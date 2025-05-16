from dataclasses import dataclass
from typing import List

from .EditorConfig import EditorConfig


@dataclass
class LLPConfig(EditorConfig):
    model_path: str
    prompt_path: str

    prompt_tokens_per_layer: int

    lr: float
    grad_steps: int
    kl_factor: float
    weight_decay: float
    edit_layer: int

    sample_num: int
    key_grad_steps: int
    key_lr: float
    temperature: float
    retrieval_layer: list

    device: int
