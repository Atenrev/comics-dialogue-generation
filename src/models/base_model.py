import torch

from typing import Any, OrderedDict


class BaseModel(torch.nn.Module):

    def __init__(self, config: Any) -> None:
        self.config = config

    def load_checkpoint(self, state_dict: OrderedDict) -> None:
        # Change Multi GPU to single GPU
        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("module."):
                new_key = key[len("module."):]
                state_dict[new_key] = state_dict.pop(key)

        self.load_state_dict(state_dict, strict=False)