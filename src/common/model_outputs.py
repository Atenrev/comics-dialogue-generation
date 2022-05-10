from numpy import dtype
import torch

from dataclasses import dataclass
from transformers.modeling_outputs import MultipleChoiceModelOutput


@dataclass
class TextClozeModelOutput(MultipleChoiceModelOutput):
    prediction: torch.Tensor = None

    def __post_init__(self):
        self.prediction = torch.argmax(self.logits, dim=1)