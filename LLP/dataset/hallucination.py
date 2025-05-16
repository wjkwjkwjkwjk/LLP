import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


class HallucDataset:

    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        with open(data_dir, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):

            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": [record["prompt"]],
                        "subject": record["subject"],
                        "target_new": [record["target_new"]],
                    },
                    "neighborhood_prompts": {
                        "prompt": record["locality_prompt"] + " ",
                        "target": record["locality_ground_truth"],
                    },
                }
            )

        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
