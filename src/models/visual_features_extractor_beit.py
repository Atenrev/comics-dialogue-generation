import torch

from typing import Any
from torch import nn
from transformers import BeitModel

from src.models.base_model import BaseModel


class VisualFeaturesExtractorBeit(BaseModel):
    def __init__(self, config: Any) -> None:
        super(VisualFeaturesExtractorBeit, self).__init__(config)
        self.beit = BeitModel.from_pretrained(config.architecture)

    def forward(self, image: dict) -> torch.Tensor:
        embedding = self.beit(**image)
        return embedding
