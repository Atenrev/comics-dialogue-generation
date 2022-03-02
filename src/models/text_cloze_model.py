import torch
from typing import Any
from torch import nn
from transformers.modeling_outputs import MultipleChoiceModelOutput

from src.modules.encoders import BaseT5EncoderModule


class TextClozeTextOnlyModel(nn.Module):

    def __init__(self, config: Any) -> None:
        super(TextClozeTextOnlyModel, self).__init__()
        self.num_labels = config.answer_candidates
        self.loss_function = nn.CrossEntropyLoss()

        # self.answers_embedding = nn.Embedding(
        #     config.num_tokens, config.answer_embed_size)
        self.t5_encoder = BaseT5EncoderModule(config)
        self.embedding = self.t5_encoder.encoder.shared
        self.fc1 = nn.Sequential(nn.Linear(90*512, 512), nn.LeakyReLU())
        self.dropout = nn.Dropout(config.dropout)

        # self.scores_fc = nn.Linear(
        #     config.answer_embed_size * config.answer_candidates *
        #     config.answer_max_tokens + config.pooler_size,
        #     config.answer_candidates
        # )
        self.scores_fc = nn.Linear(
            config.pooler_size,
            config.answer_candidates
        )

    def forward(self,
                context: torch.Tensor,
                answers: torch.Tensor,
                targets: torch.Tensor) -> MultipleChoiceModelOutput:
        # inp = torch.cat((context, answers), 1)
        context_encoding_outputs = self.t5_encoder(context)
        context_encoding_outputs = self.dropout(context_encoding_outputs)
        answer_embedding_outputs = self.embedding(answers)
        answer_embedding_outputs = self.fc1(answer_embedding_outputs.view(answer_embedding_outputs.size(0), -1))
        answer_embedding_outputs = self.dropout(answer_embedding_outputs)
        # answer_encoding_outputs = self.t5_encoder(answers)
        # .view(outputs.size(0), -1)
        # outputs = torch.cat(
        #     (context_encoding_outputs, answer_embedding_outputs), 1)
        outputs = context_encoding_outputs + answer_embedding_outputs
        logits = self.scores_fc(outputs)
        loss = None

        if targets is not None:
            loss = self.loss_function(logits, targets)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )
