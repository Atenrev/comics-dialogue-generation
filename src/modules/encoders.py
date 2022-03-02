from typing import Any
import torch
from torch import nn
from transformers import T5EncoderModel

from src.modules.poolers import T5Pooler


class BaseT5EncoderModule(nn.Module):

    def __init__(self, config: Any) -> None:
        super(BaseT5EncoderModule, self).__init__()
        # Should we freeze the T5?
        self.encoder = T5EncoderModel.from_pretrained("t5-small")
        self.pooler = T5Pooler(config)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(context)
        pooler_outputs = self.pooler(encoder_outputs.last_hidden_state)
        return pooler_outputs
