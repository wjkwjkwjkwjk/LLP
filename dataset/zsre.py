import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


class ZsREDataset:

    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        with open(data_dir, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            assert (
                "nq question: " in record["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
            while record["alt"][0] == " ":
                record["alt"] = record["alt"][1:]

            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": [record["src"]],
                        "subject": record["subject"],
                        "target_new": [record["alt"]],
                        "target_true": record["answers"][0],
                    },
                    "paraphrase_prompts": record["rephrase"],
                    "neighborhood_prompts": {
                        "prompt": record["loc"].replace("nq question: ", "") + "?",
                        "target": record["loc_ans"],
                    },
                }
            )

        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
