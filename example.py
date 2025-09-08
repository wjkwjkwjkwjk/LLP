from editor.LLP.LLP import LLP
from dataset.zsre import ZsREDataset
from dataset.hallucination import HallucDataset
from collections import defaultdict


requests = [
    {
        "case_id": 0,
        "requested_rewrite": {
            "prompt": ["What university did Watts Humphrey attend?"],
            "subject": "Watts Humphrey",
            "target_new": ["University of Michigan"],
        },
        "paraphrase_prompts": "What university did Watts Humphrey take part in?",
        "neighborhood_prompts": {
            "prompt": "who played desmond doss father in hacksaw ridge",
            "target": "Hugo Weaving",
        },
    },
    # {...},
    # {...},
]
config_path = "LLP/evaluation/LLP/llama_3_8b.json"
editor = LLP(config_path)
editor.edit(requests)
model=editor.get_model()
tokenizer=editor.get_tokenizer()

prompts = ["What university did Watts Humphrey attend?"]
tok =tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
            ).to(0)
prd_rel = model.forward_with_memory(**tok)