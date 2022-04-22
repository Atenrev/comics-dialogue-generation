import torch

from typing import Any
from torch import nn
from transformers import BeitModel


class VisualFeaturesExtractorBeit(nn.Module):
    def __init__(self, config: Any) -> None:
        super(VisualFeaturesExtractorBeit, self).__init__()
        self.config = config
        self.beit = BeitModel.from_pretrained(config.architecture)

    def forward(self, image: dict, image_id: str) -> torch.Tensor:
        embedding = self.beit(**image)
        return embedding
