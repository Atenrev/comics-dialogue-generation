import torch
from typing import Any
from torch import nn
from transformers.modeling_outputs import MultipleChoiceModelOutput

from src.modules.encoders import BaseT5EncoderModule
from src.modules.poolers import MeanPooler


class TextClozeImageTextT5Model(nn.Module):

    def __init__(self, config: Any) -> None:
        super(TextClozeImageTextT5Model, self).__init__()
        self.config = config
        self.num_labels = config.answer_candidates
        self.loss_function = nn.CrossEntropyLoss()
        self.images_pooler = MeanPooler(config)
        self.encoder = BaseT5EncoderModule(config)
        self.dropout = nn.Dropout(config.dropout)
        self.scores_fc = nn.Linear(
            config.pooler_size,
            config.answer_candidates
        )

    def forward(
        self,
        dialogues: torch.Tensor,
        images: torch.Tensor,
        answers: torch.Tensor,
        targets: torch.Tensor
    ) -> MultipleChoiceModelOutput:
        """
        Args:
            dialogues: [batch_size, max_dialogue_length, max_dialogue_length]
            images: [batch_size, max_panels, 197, 768]
            answers: [batch_size, max_dialogue_length, max_dialogue_length]
            targets: [batch_size]

        Returns:
            loss: [batch_size]
            logits: [batch_size, num_labels]
        """
        batch_size = dialogues.size(0)
        
        # Images are not to be embedded
        images = self.images_pooler(images)

        joint = torch.cat(
            (dialogues.view(batch_size, -1),
            images.view(batch_size, -1),
            answers.view(batch_size, -1)),
            1
        )

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
