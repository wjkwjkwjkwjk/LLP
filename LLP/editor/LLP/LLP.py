from transformers import AutoModelForCausalLM, AutoTokenizer
from .Prompt_model import PromptLlama, PromptGPTJ, PromptMistral
from transformers.models.llama import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
import torch
import random
import torch.nn.functional as F
from torch import nn
from utils.config.LLPConfig import LLPConfig
from utils.log.LLPLog import LLPLog
from utils.utils import (
    set_requires_grad,
    generate_fast,
    get_subject_idxs_in_prompts,
    cosine_similarity_between_batches,
)
import os
from pathlib import Path
import json
from collections import defaultdict
import gc
import time 


def copy_file(src_file, dest_file):
    with open(src_file, "r") as src, open(dest_file, "w") as dest:
        dest.write(src.read())
        src.close
        dest.close


def generate_mask(request_len, gen_len):
    row_indices = torch.arange(request_len).unsqueeze(1)
    col_indices = torch.arange(gen_len * request_len).unsqueeze(0)
    start_indices = row_indices * gen_len
    end_indices = (row_indices + 1) * gen_len
    mask = (col_indices >= start_indices) & (col_indices < end_indices)
    return mask


def concat_tensors(a, b, dim):
    if a is not None and b is not None:
        return torch.cat((a, b), dim=dim)
    elif a is not None:
        return a
    elif b is not None:
        return b
    else:
        return None


class LLP:
    def __init__(
        self,
        config_path: str,
    ):
        self.config = LLPConfig.from_json(config_path)
        self.device = self.config.device
        if "Llama" in self.config.model_path:
            self.model = PromptLlama.from_pretrained(
                self.config.model_path, low_cpu_mem_usage=True  
            ).to(self.device)
        elif "gpt-j" in self.config.model_path:
            self.model = PromptGPTJ.from_pretrained(
                self.config.model_path, low_cpu_mem_usage=True
            ).to(self.device)
        elif "Mistral" in self.config.model_path:
            self.model = PromptMistral.from_pretrained(
                self.config.model_path, low_cpu_mem_usage=True
            ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.prompt_path = Path(self.config.prompt_path)
        self.model.prompt_tokens_per_layer = self.config.prompt_tokens_per_layer
        self.sample_num = self.config.sample_num
        self.model.edit_layer = self.config.edit_layer
        self.model.retrieval_layer = self.config.retrieval_layer
        self.log_path = self.prompt_path / "log.json"
        self.prompt_memory_path = self.prompt_path / "prompt_memory.pth"
        self.subject_key_memory_path = self.prompt_path / "subject_key_memory.pth"
        self.model.prompt_value = None
        self.model.prompt_value = None
        self.model.subject_key = None

        if os.path.exists(self.prompt_path):
            if os.path.exists(self.log_path):
                self.log = LLPLog.from_json(self.log_path)
            else:
                log = {
                "model_path": self.config.model_path,
                "prompt_path": self.config.prompt_path,
                "prompt_tokens_per_layer": self.config.prompt_tokens_per_layer,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "kl_factor": self.config.kl_factor,
                "grad_steps": self.config.grad_steps,
                "device": self.config.device,
                "edit_num": 0,
                "sample_num": self.config.sample_num,
                "key_grad_steps": self.config.key_grad_steps,
                "key_lr": self.config.key_lr,
                "temperature": self.config.temperature,
                "edit_layer": self.config.edit_layer,
                "retrieval_layer": self.config.retrieval_layer,
            }
                with open(self.log_path, "w") as f:
                    json.dump(log, f, indent=4)
            self.log = LLPLog.from_json(self.log_path)
            assert self.log.edit_layer==self.config.edit_layer and self.log.retrieval_layer==self.config.retrieval_layer,"The retrieval layer and edit layer must remain consistent with previous settings."
            if os.path.exists(self.prompt_memory_path):
                self.model.prompt_value = torch.load(self.prompt_memory_path).to(
                    self.device
                )
            if os.path.exists(self.subject_key_memory_path):
                self.model.subject_key = torch.load(self.subject_key_memory_path).to(
                    self.device
                )
            else:
                self.log.edit_num = 0

        else:
            os.mkdir(self.prompt_path)
            log = {
                "model_path": self.config.model_path,
                "prompt_path": self.config.prompt_path,
                "prompt_tokens_per_layer": self.config.prompt_tokens_per_layer,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "kl_factor": self.config.kl_factor,
                "grad_steps": self.config.grad_steps,
                "device": self.config.device,
                "edit_num": 0,
                "sample_num": self.config.sample_num,
                "key_grad_steps": self.config.key_grad_steps,
                "key_lr": self.config.key_lr,
                "temperature": self.config.temperature,
                "edit_layer": self.config.edit_layer,
                "retrieval_layer": self.config.retrieval_layer,
            }
            with open(self.log_path, "w") as f:
                json.dump(log, f, indent=4)
            self.log = LLPLog.from_json(self.log_path)

    def edit(self, requests):
        self.get_value(requests)
        self.get_key(requests)
        torch.save(self.model.prompt_value, self.prompt_memory_path)
        torch.save(self.model.subject_key, self.subject_key_memory_path)
        add_num=len(requests)
        self.log.edit_num+=add_num
        log = {
                "model_path": self.config.model_path,
                "prompt_path": self.config.prompt_path,
                "prompt_tokens_per_layer": self.config.prompt_tokens_per_layer,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "kl_factor": self.config.kl_factor,
                "grad_steps": self.config.grad_steps,
                "device": self.config.device,
                "edit_num": self.log.edit_num,
                "sample_num": self.config.sample_num,
                "key_grad_steps": self.config.key_grad_steps,
                "key_lr": self.config.key_lr,
                "temperature": self.config.temperature,
                "edit_layer": self.config.edit_layer,
                "retrieval_layer": self.config.retrieval_layer,
            }
        with open(self.log_path, "w") as f:
            json.dump(log, f, indent=4)

    def get_value(self, requests):
        vtimes=[]
        n_gen_per_prompt = 3
        prefix_prompts = generate_fast(
            self.model,
            self.tokenizer,
            ["<|endoftext|>"],
            top_k=5,
            n_gen_per_prompt=n_gen_per_prompt,
            max_out_len=15,
        )
        print("Start getting memory values")
        for req_id, request in enumerate(requests):
            vstart_time = time.time()
            rel_prompts = request["requested_rewrite"]["prompt"]
            targets_new = request["requested_rewrite"]["target_new"]
            subject = request["requested_rewrite"]["subject"]
            targets_knowledge = [
                rel_prompt + " " + target_new
                for rel_prompt, target_new in zip(rel_prompts, targets_new)
            ]

            print(f"subject:{subject}")
            for rel_prompt, target_new in zip(rel_prompts, targets_new):
                print(f"{rel_prompt}------->{target_new}")

            new_prompt_value = torch.randn(
                [
                    self.model.prompt_tokens_per_layer,
                    self.model.hidden_size,
                ],
                device=self.device,
            ).detach()

            ############### initializing soft_prompt
            targets_knowledge_tok = self.tokenizer(
                targets_knowledge,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                target_knowledge_hidden_states = self.model(
                    **targets_knowledge_tok, output_hidden_states=True
                ).hidden_states

            new_prompt_value = target_knowledge_hidden_states[
                self.model.edit_layer
            ].mean(0)[-self.model.prompt_tokens_per_layer :]
            new_prompt_value_init = new_prompt_value.detach()

            ######### Generate training data.
            # loc_prompts = self.tokenizer.decode(
            #     [random.randint(0, self.tokenizer.vocab_size)]
            # )

            # loc_prompt_ids = self.tokenizer(loc_prompts, return_tensors="pt").to(
            #     self.device
            # )["input_ids"]
            # loc_prompts = self.tokenizer.batch_decode(
            #     loc_prompt_ids,
            #     skip_special_tokens=True,
            #     clean_up_tokenization_spaces=False,
            # )
            # loc_prompts = generate_fast(
            #     self.model,
            #     self.tokenizer,
            #     loc_prompts,
            #     top_k=5,
            #     n_gen_per_prompt=10,
            #     max_out_len=10,
            # )
            # loc_prompts = [
            #     prefix_prompt + " " + subject + " is "
            #     for prefix_prompt in prefix_prompts
            # ]

            rel_prompts = [
                prefix_prompt + " " + rel_prompt
                for rel_prompt in rel_prompts
                for prefix_prompt in prefix_prompts
            ]
            rel_prompts_tok = self.tokenizer(
                rel_prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            rel_prompts_tokompt_len = rel_prompts_tok["attention_mask"].sum(1)
            targets_knowledge = [
                prefix_prompt + " " + target_knowledge
                for target_knowledge in targets_knowledge
                for prefix_prompt in prefix_prompts
            ]
            targets_knowledge_tok = self.tokenizer(
                targets_knowledge,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            targets_knowledge_len = targets_knowledge_tok["attention_mask"].sum(1)
            mask = (
                torch.tensor(1)
                .repeat(
                    targets_knowledge_tok["input_ids"].shape[0],
                    targets_knowledge_tok["input_ids"].shape[1] - 1,
                )
                .to(self.device)
            )
            for i in range(len(targets_knowledge)):
                mask[i, : rel_prompts_tokompt_len[i] - 1] = 0
                mask[i, targets_knowledge_len[i] - 1 :] = 0
            target_size = mask.sum(1)
            loc_init = None
            ############## starting traininng
            new_prompt_value.requires_grad_()
            opt = torch.optim.Adam([new_prompt_value], lr=self.config.lr)
            if (
                "Llama" in self.config.model_path
                or "Mistral" in self.config.model_path
            ):
                set_requires_grad(False, self.model.model)
                set_requires_grad(False, self.model.lm_head)
            elif "gpt-j" in self.config.model_path:
                set_requires_grad(False, self.model.transformer)
                set_requires_grad(False, self.model.lm_head)
            for it in range(self.config.grad_steps):
                opt.zero_grad()
                output = self.model.forward_with_single_prompt(
                    **targets_knowledge_tok,
                    soft_prompt=new_prompt_value,
                    output_hidden_states=True,
                    edit_layer=self.model.edit_layer,
                )

                logits = output.logits
                log_probs = torch.log_softmax(logits, dim=2)
                if loc_init is None:
                    loc_init = log_probs.detach().clone()
                ############## Calculate reliability loss
                log_probs = torch.gather(
                    log_probs[:, :-1, :],
                    2,
                    targets_knowledge_tok["input_ids"][:, 1:, None],
                ).squeeze(2)

                rel_loss = -(log_probs * mask).sum(1) / target_size
                rel_loss = rel_loss.mean()

                ############## Calculate locality loss
                loc_loss = 0
                # loc_loss = self.config.kl_factor * torch.nn.functional.kl_div(
                #     loc_init[len(rel_prompts) :, :, :],
                #     log_probs[len(rel_prompts) :, :, :],
                #     log_target=True,
                #     reduction="batchmean",
                # )
                l2_loss = 0
                l2_loss = self.config.weight_decay * (
                    torch.norm(new_prompt_value)
                    / torch.norm(new_prompt_value_init) ** 2
                )

                loss = rel_loss + loc_loss + l2_loss
                if (it + 1) % 5 == 0:
                    # if True:
                    print(f"epoch{it+1}")
                    print(f"rel_loss:{rel_loss}  l2_loss:{l2_loss}")
                    print(f"loss:{loss}")

                loss.backward()
                opt.step()
                if loss < 5e-2:
                    break

            if req_id == 0:
                prompt_value = new_prompt_value.unsqueeze(0)
            else:
                prompt_value = torch.cat(
                    (prompt_value, new_prompt_value.unsqueeze(0)), dim=0
                )
            vend_time = time.time()
            vtime=vend_time-vstart_time
            vtimes.append(vtime)
            print(f"time cost: {vtime}")
        # torch.save(prompt_value, self.prompt_memory_path)
        if self.model.prompt_value == None:
            self.model.prompt_value = prompt_value
        else:
            self.model.prompt_value = torch.cat(
                (self.model.prompt_value, prompt_value), dim=0
            )
        # torch.save(self.model.prompt_value, self.prompt_memory_path)
        return prompt_value

    def get_key(self, requests):
        n_gen_per_prompt = 3
        prefix_prompts = generate_fast(
            self.model,
            self.tokenizer,
            ["<|endoftext|>"],
            top_k=5,
            n_gen_per_prompt=n_gen_per_prompt,
            max_out_len=15,
        )
        ktimes=[]
        print("Starting getting memory keys")
        for req_id, request in enumerate(requests):
            kstart_time=time.time()
            rel_prompts = request["requested_rewrite"]["prompt"]
            subject = request["requested_rewrite"]["subject"]
            targets_new = request["requested_rewrite"]["target_new"]
            print(f"subject:{subject}")
            for rel_prompt, target_new in zip(rel_prompts, targets_new):
                print(f"{rel_prompt}------->{target_new}")

            retrieval_layer_num = len(self.model.retrieval_layer)
            new_subject_key = torch.randn(
                [retrieval_layer_num, 1, self.model.hidden_size], device=self.device
            ).detach()
            # new_relation_key = torch.randn(
            #     [retrieval_layer_num, 1, self.model.hidden_size], device=self.device
            # ).detach()

            query_prompts = [
                prefix_prompt + rel_prompt
                for rel_prompt in rel_prompts
                for prefix_prompt in prefix_prompts
            ]
            query_prompts_tok = self.tokenizer(
                query_prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            subject_idxs = get_subject_idxs_in_prompts(
                self.tokenizer, query_prompts, subject
            )
            subject_idxs = torch.tensor(
                subject_idxs, dtype=torch.long, device=self.device
            )
            query_last_token_idxs = (
                torch.sum(query_prompts_tok["attention_mask"], dim=1) - 1
            )

            with torch.no_grad():
                query_prompt_hidden_states = self.model(
                    **query_prompts_tok, output_hidden_states=True
                ).hidden_states

            ########## getting query
            subject_query = torch.randn(
                [
                    retrieval_layer_num,
                    n_gen_per_prompt * len(rel_prompts),
                    self.model.hidden_size,
                ],
                device=self.device,
            ).detach()
            # relation_query = torch.randn(
            #     [retrieval_layer_num, gen_len + 1, self.model.hidden_size],
            #     device=self.device,
            # ).detach()

            for index, layer_id in enumerate(self.model.retrieval_layer):
                subject_query[index] = query_prompt_hidden_states[layer_id+1][
                    torch.arange(len(query_prompts)), subject_idxs
                ]
                # relation_query[index] = query_prompt_hidden_states[layer_id][
                #     torch.arange(len(query_prompts)), query_last_token_idxs
                # ]
                ############### initializing key
                new_subject_key[index][0] = subject_query[index][-1]
                # new_relation_key[index][0] = relation_query[index][-1]

            new_subject_key.requires_grad_()
            # new_relation_key.requires_grad_()

            if req_id == 0 and self.log.edit_num == 0:
                subject_key = new_subject_key
                # relation_key = new_relation_key
                continue
            ###compute new_subject_key
            print("compute new_subject_key")
            opt = torch.optim.Adam([new_subject_key], lr=self.config.key_lr)
            if (
                "Llama" in self.config.model_path
                or "Mistral" in self.config.model_path
            ):
                set_requires_grad(False, self.model.model)
                set_requires_grad(False, self.model.lm_head)
            elif "gpt-j" in self.config.model_path:
                set_requires_grad(False, self.model.transformer)
                set_requires_grad(False, self.model.lm_head)
            for index, laye_id in enumerate(self.model.retrieval_layer):
                for it in range(self.config.key_grad_steps):
                    opt.zero_grad()
                    if req_id != 0:
                        old_subject_key_1 = subject_key
                    else:
                        old_subject_key_1 = None
                    if self.log.edit_num != 0:
                        old_subject_key_2 = self.model.subject_key
                        # [
                        #     :,
                        #     torch.randperm(self.log.edit_num)[: self.sample_num],
                        #     :,
                        # ]
                    else:
                        old_subject_key_2 = None
                    old_subject_key = concat_tensors(
                        old_subject_key_1, old_subject_key_2, 1
                    )
                    if old_subject_key!=None:
                        old_subject_key=old_subject_key[
                            :,
                            torch.randperm(old_subject_key.shape[1])[: self.sample_num],
                            :,
                        ]
                    positive_exp = torch.exp(
                        cosine_similarity_between_batches(
                            new_subject_key[index],
                            subject_query[index],
                        )
                        / self.config.temperature
                    ).mean()
                    negative_exp = torch.exp(
                        cosine_similarity_between_batches(
                            new_subject_key[index],
                            old_subject_key[index],
                        )
                        / self.config.temperature
                    ).mean()
                    loss = -torch.log(positive_exp / (positive_exp + negative_exp))
                    # if (it + 1) % 5 == 0:
                    #     print(f"epoch{it+1}")
                    #     print(f"loss:{loss}")
                    loss.backward()
                    if loss < 5e-2:
                        break
                    opt.step()

            ###compute new_relation_key
            # print("compute new_relation_key")
            # opt = torch.optim.Adam([new_relation_key], lr=self.config.key_lr)
            # set_requires_grad(False, self.model.model)
            # for index, laye_id in enumerate(self.model.retrieval_layer):
            #     # print(f"layer{laye_id}")
            #     for it in range(self.config.key_grad_steps):
            #         opt.zero_grad()
            #         if req_id != 0:
            #             old_relation_key_1 = relation_key
            #         else:
            #             old_relation_key_1 = None
            #         if self.log.edit_num != 0:
            #             old_relation_key_2 = self.model.relation_key[
            #                 :,
            #                 torch.randperm(self.log.edit_num)[: self.sample_num],
            #                 :,
            #             ]
            #         else:
            #             old_relation_key_2 = None
            #         old_relation_key = concat_tensors(
            #             old_relation_key_1, old_relation_key_2, 1
            #         )
            #         positive_exp = torch.exp(
            #             cosine_similarity_between_batches(
            #                 new_relation_key[index], relation_query[index]
            #             )
            #             / self.config.temperature
            #         ).mean()
            #         negative_exp = torch.exp(
            #             cosine_similarity_between_batches(
            #                 new_relation_key[index], old_relation_key[index]
            #             )
            #             / self.config.temperature
            #         ).mean()
            #         loss = -torch.log(positive_exp / (positive_exp + negative_exp))
            #         # if (it + 1) % 5 == 0:
            #         #     print(f"epoch{it+1}")
            #         #     print(f"loss:{loss}")
            #         loss.backward()
            #         if loss < 0.05:
            #             break
            #         opt.step()

            if req_id == 0:
                subject_key = new_subject_key
                # relation_key = new_relation_key
            else:
                subject_key = torch.cat((subject_key, new_subject_key), dim=1)
                # relation_key = torch.cat((relation_key, new_relation_key), dim=1)
            kend_time=time.time()
            ktime=kend_time-kstart_time
            ktimes.append(ktime)
            print(f"time cost:{ktime}")
        
        if self.model.subject_key == None:
            self.model.subject_key = subject_key
        else:
            self.model.subject_key = torch.cat(
                (self.model.subject_key, subject_key), dim=1
            )
        # torch.save(self.model.subject_key, self.subject_key_memory_path)
        # with open("ktimes_G.json", "w") as f:
        #     json.dump(ktimes, f)
        return subject_key

    def evaluate(self, requests):
        all_rel_acc = 0
        all_paraphrase_acc = 0
        all_loc_acc = 0
        pre_times=[]
        post_times=[]
        for req_id, request in enumerate(requests):
            rel_prompts = request["requested_rewrite"]["prompt"]
            paraphrase_prompt = request["paraphrase_prompts"]
            paraphrase_prompts = [paraphrase_prompt]
            target_new = request["requested_rewrite"]["target_new"][0]
            loc_prompt = request["neighborhood_prompts"]["prompt"]
            loc_prompts = [loc_prompt]
            loc_target = request["neighborhood_prompts"]["target"]

            rel_prompts_tok = self.tokenizer(
                rel_prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            rel_prompts_len = rel_prompts_tok["attention_mask"].sum(1)
            rel_prompts = [prompt + " " + target_new for prompt in rel_prompts]
            rel_prompts_tok = self.tokenizer(
                rel_prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            paraphrase_prompt_tok = self.tokenizer(
                paraphrase_prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            paraphrase_prompt_len = paraphrase_prompt_tok["attention_mask"].sum(1)
            paraphrase_prompts = [
                prompt + " " + target_new for prompt in paraphrase_prompts
            ]
            paraphrase_prompt_tok = self.tokenizer(
                paraphrase_prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            loc_target_ids = self.tokenizer(loc_target, return_tensors="pt").to(
                self.device
            )["input_ids"][0]
            if (
                loc_target_ids[0] == self.tokenizer.bos_token_id
                or loc_target_ids[0] == self.tokenizer.unk_token_id
            ):
                loc_target_ids = loc_target_ids[1:]

            loc_prompts = [
                prompt + " " + self.tokenizer.decode(loc_target_ids[:-1])
                for prompt in loc_prompts
            ]
            loc_prompts_tok = self.tokenizer(
                loc_prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                pre_start=time.time()
                pre_prd_loc = self.model.forward(**loc_prompts_tok).logits
                pre_end=time.time()
                pre_time=pre_end-pre_start
                pre_times.append(pre_time)
                pre_prd_loc = torch.softmax(pre_prd_loc, 2).argmax(2)
                post_start=time.time()
                prd_loc = self.model.forward_with_memory(
                    **loc_prompts_tok,
                ).logits
                post_end=time.time()
                post_time=post_end-post_start
                post_times.append(post_time)
                prd_loc = torch.softmax(prd_loc, 2).argmax(2)
                prd_loc = prd_loc[:, -len(loc_target_ids) :]
                pre_prd_loc = pre_prd_loc[:, -len(loc_target_ids) :]
                loc_acc = (prd_loc == pre_prd_loc).sum(1) / loc_target_ids.shape[0]
                pre_loc_prd = [
                    [
                        self.tokenizer.decode(pre_prd_loc[i][j])
                        for j in range(len(pre_prd_loc[i]))
                    ]
                    for i in range(len(pre_prd_loc))
                ]
                loc_prd = [
                    [
                        self.tokenizer.decode(prd_loc[i][j])
                        for j in range(len(prd_loc[i]))
                    ]
                    for i in range(len(prd_loc))
                ]
                all_loc_acc += loc_acc

                prd_rel = self.model.forward_with_memory(**rel_prompts_tok)
                prd_rel = prd_rel.logits
                prd_rel = torch.softmax(prd_rel, 2).argmax(2)
                prd_rel = prd_rel[:, rel_prompts_len[0] - 1 : -1]
                target_size = prd_rel.shape[1]
                rel_acc = (
                    prd_rel == rel_prompts_tok["input_ids"][:, -target_size:]
                ).sum(1) / target_size
                rel_prd = [
                    [
                        self.tokenizer.decode(prd_rel[i][j])
                        for j in range(len(prd_rel[i]))
                    ]
                    for i in range(len(prd_rel))
                ]
                all_rel_acc += rel_acc[0]

                prd_paraphrase = self.model.forward_with_memory(
                    **paraphrase_prompt_tok,
                )
                # print(prd_paraphrase.retrieval_result)
                prd_paraphrase = prd_paraphrase.logits
                prd_paraphrase = torch.softmax(prd_paraphrase, 2).argmax(2)
                prd_paraphrase = prd_paraphrase[:, paraphrase_prompt_len[0] - 1 : -1]
                target_size = prd_paraphrase.shape[1]
                paraphrase_acc = (
                    prd_paraphrase
                    == paraphrase_prompt_tok["input_ids"][:, -target_size:]
                ).sum(1) / target_size
                paraphrase_prd = [
                    [
                        self.tokenizer.decode(prd_paraphrase[i][j])
                        for j in range(len(prd_paraphrase[i]))
                    ]
                    for i in range(len(prd_paraphrase))
                ]
                all_paraphrase_acc += paraphrase_acc[0]

            #     rt = {
            #         "target": target_new,
            #         "rel_prompt": rel_prompts[0],
            #         "paraphrase_prompt": paraphrase_prompt,
            #         "loc_prompt": loc_prompt,
            #         "rel_predict": rel_prd,
            #         "paraphrase_predict": paraphrase_prd,
            #         "rel_acc": rel_acc.cpu().detach().cpu().numpy().tolist(),
            #         "paraphrase_acc": paraphrase_acc.cpu()
            #         .detach()
            #         .cpu()
            #         .numpy()
            #         .tolist(),
            #         "pre_loc_predict": pre_loc_prd,
            #         "loc_predict": loc_prd,
            #         "local_acc": loc_acc.cpu().detach().cpu().numpy().tolist(),
            #     }
            # with open(self.prompt_path / f"{req_id}.json", "w") as f:
            #     json.dump(rt, f, indent=4)
        # print(all_rel_acc / len(requests))
        # print(all_paraphrase_acc / len(requests))
        # print(all_loc_acc / len(requests))

        return (
            all_rel_acc / len(requests),
            all_paraphrase_acc / len(requests),
            all_loc_acc / len(requests),
        )

    def evaluate_ppl(self, requests):
        all_loc_acc = 0
        all_ppl = 0
        for req_id, request in enumerate(requests):
            rel_prompt = request["requested_rewrite"]["prompt"][0]
            target_new = request["requested_rewrite"]["target_new"][0]
            loc_prompt = request["neighborhood_prompts"]["prompt"]
            loc_target = request["neighborhood_prompts"]["target"]
            loc_prompts = [loc_prompt]
            query_prompt = rel_prompt + " " + target_new
            prompt_len = len(
                self.tokenizer([rel_prompt], return_tensors="pt")["input_ids"][0]
            )
            query_prompt_tok = self.tokenizer(
                [query_prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                output = self.model.forward_with_memory(**query_prompt_tok)
            logits = output.logits
            log_probs = torch.log_softmax(logits, dim=2)
            a = torch.argmax(log_probs, dim=2)
            log_probs = torch.gather(
                log_probs[:, :-1, :], 2, query_prompt_tok["input_ids"][:, 1:, None]
            )[0]

            log_probs = log_probs[prompt_len - 1 :].squeeze(1)
            ppl = torch.exp(-log_probs.mean()).item()
            all_ppl += ppl

            if loc_target == "":
                all_loc_acc += 1
                continue
            loc_target_ids = self.tokenizer(loc_target, return_tensors="pt").to(
                self.device
            )["input_ids"][0]
            if (
                loc_target_ids[0] == self.tokenizer.bos_token_id
                or loc_target_ids[0] == self.tokenizer.unk_token_id
            ):
                loc_target_ids = loc_target_ids[1:]

            loc_prompts = [
                prompt + " " + self.tokenizer.decode(loc_target_ids[:-1])
                for prompt in loc_prompts
            ]
            loc_prompts_tok = self.tokenizer(
                loc_prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                pre_prd_loc = self.model.forward(
                    **loc_prompts_tok, output_hidden_states=True
                )
                pre_prd_loc = pre_prd_loc.logits
                pre_prd_loc = torch.softmax(pre_prd_loc, 2).argmax(2)

                prd_loc = self.model.forward_with_memory(
                    **loc_prompts_tok, output_hidden_states=True
                )
                prd_loc = prd_loc.logits
                prd_loc = torch.softmax(prd_loc, 2).argmax(2)
                prd_loc = prd_loc[:, -len(loc_target_ids) :]
                pre_prd_loc = pre_prd_loc[:, -len(loc_target_ids) :]
                loc_acc = (prd_loc == pre_prd_loc).sum(1) / loc_target_ids.shape[0]

                all_loc_acc += loc_acc
                # print(loc_acc)
        # print(all_ppl / len(requests))
        # print(all_loc_acc / len(requests))
        return all_ppl / len(requests), all_loc_acc / len(requests)

    def unload_model(self):
        del self.model
        torch.cuda.empty_cache()  
        gc.collect()  

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
    
    def delete_memory(self):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        if os.path.exists(self.prompt_memory_path):
            os.remove(self.prompt_memory_path)
        if os.path.exists(self.subject_key_memory_path):
            os.remove(self.subject_key_memory_path)
        self.model.prompt_value = None
        self.model.subject_key = None
        self.log.edit_num=0
