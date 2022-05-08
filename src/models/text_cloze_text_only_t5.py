import torch
from typing import Any
from torch import nn
from transformers.modeling_outputs import MultipleChoiceModelOutput

from src.models.base_model import BaseModel
from src.modules.encoders import BaseT5EncoderModule


class TextClozeTextOnlyT5Model(BaseModel):

    def __init__(self, config: Any) -> None:
        super(TextClozeTextOnlyT5Model, self).__init__(config)
        self.num_labels = config.answer_candidates
        self.loss_function = nn.CrossEntropyLoss()
        self.encoder = BaseT5EncoderModule(config)
        self.dropout = nn.Dropout(config.dropout)
        self.scores_fc = nn.Linear(
            config.pooler_size,
            config.answer_candidates
        )

    def forward(self,
                context: torch.Tensor,
                answers: torch.Tensor,
                targets: torch.Tensor) -> MultipleChoiceModelOutput:
        batch_size = context.size(0)
        joint = torch.cat((context.view(batch_size, -1), answers.view(batch_size, -1)), 1)
        context_encoding_outputs = self.encoder(joint)
        context_encoding_outputs = self.dropout(context_encoding_outputs)
        logits = self.scores_fc(context_encoding_outputs)
        loss = None

        if targets is not None:
            loss = self.loss_function(logits, targets)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
        )
