from editor.LLP.LLP import LLP
from dataset.zsre import ZsREDataset
from dataset.hallucination import HallucDataset
from collections import defaultdict
import shutil
import torch
import os

###Used for the hallucination dataset, where multiple edits are performed on the same subject.
def merge_requests(requests):
    merged_data = defaultdict(list)
    for item in requests:
        merged_data[item["requested_rewrite"]["subject"]].append(item)
    requests = []
    for subject, items in merged_data.items():
        requests.append(
            {
                "requested_rewrite": {
                    "prompt": [
                        record["requested_rewrite"]["prompt"][0] for record in items
                    ],
                    "subject": subject,
                    "target_new": [
                        record["requested_rewrite"]["target_new"][0] for record in items
                    ],
                },
            }
        )
    return requests

def evaluate(dataset,editing_size,config_path):
    # dataset = "QA"### hallucination
    n = editing_size 
    # config_path = "./evaluation/LLP/llama_3_8b.json"


    if dataset == "QA":
        data_path = "./dataset/data/new_zsre_mend_edit.json"
        dst = ZsREDataset(data_path)
    elif dataset == "hallucination":
        data_path = "./dataset/data/new_hallucination-edit.json"
        dst = HallucDataset(data_path)

    all_ppl = 0
    all_loc = 0
    all_rel = 0
    all_para = 0
    step = int(len(dst) / n)
    editor = LLP(config_path)
    print(f"Total number of samples:{len(dst)},editing size(n):{n},total number of step:{step}")
    for i in range(step):
        print(f"step:{i}")
        requests = []
        for id, sample in enumerate(dst):
            if id >= (i + 1) *n:
                break
            if id >= i * n :
                requests.append(sample)
        if dataset == "hallucination":
            original_requests = requests
            requests = merge_requests(requests)
        editor.edit(requests)

        
        if dataset == "QA":
            rel, para, loc = editor.evaluate(
                requests
            )
            all_rel += rel
            all_para += para
            all_loc += loc
        if dataset == "hallucination":
            ppl, loc = editor.evaluate_ppl(original_requests)
            all_ppl += ppl
            all_loc += loc
        editor.delete_memory()

    if dataset == "QA":
        print(f"rel:{all_rel / step}")
        print(f"gen:{all_para / step}")
        print(f"loc:{all_loc / step}")

    if dataset == "hallucination":
        print(f"ppl:{all_ppl / step}")
        print(f"loc:{all_loc / step}")
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dst', '--dataset', type=str)
    parser.add_argument('-n', '--editing_size', type=int)
    parser.add_argument('-cf', '--config', type=str)
    args = parser.parse_args()
    print(args)
    evaluate(args.dataset, args.editing_size, args.config)
# evaluate('QA',10,'./evaluation/LLP/llama_3_8b.json')