import json
from dataclasses import dataclass


@dataclass
class EditorConfig:
    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)
            f.close

        return cls(**data)
