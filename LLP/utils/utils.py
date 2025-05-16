import unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import torch
import torch.nn.functional as F


def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
    vanilla_generation=False,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    if vanilla_generation:
        gen_txt = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_out_len,
        )
        txt = [
            tok.decode(x, skip_special_tokens=True)
            for x in gen_txt.detach().cpu().numpy().tolist()
        ]
        txt = [
            unicodedata.normalize("NFKD", x)
            .replace("\n\n", " ")
            .replace("<|endoftext|>", "")
            for x in txt
        ]
        return txt
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        # while not exceeding max output length
        while input_ids.size(1) < max_out_len:
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=(
                    None
                    if "llama" in model.name_or_path.lower()
                    or "baichuan" in model.name_or_path.lower()
                    or "internlm" in model.name_or_path.lower()
                    else attention_mask[:, cur_context]
                ),
                past_key_values=past_key_values,
                use_cache=True,
            )
            if type(model_out) is torch.Tensor:
                logits = model_out
            else:
                logits = model_out.logits
            past_key_values = model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)
    txt = [
        tok.decode(x, skip_special_tokens=True)
        for x in input_ids.detach().cpu().numpy().tolist()
    ]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]

    return txt


def get_subject_idxs_in_prompts(tok: AutoTokenizer, prompts: str, subject: str) -> int:


    idxs = []
    for i, prompt in enumerate(prompts):
        prompt = prompt.split(subject, 1)
        prefix = prompt[0] + subject
        prefix_len = len(tok.encode(prefix))
        idxs.append(prefix_len - 1)
    return idxs


def cosine_similarity_between_batches(batch_1, batch_2):
    sim_matrix = F.cosine_similarity(batch_1.unsqueeze(1), batch_2.unsqueeze(0), dim=2)
    return sim_matrix
