from typing import Any
import torch
from torch import nn
from transformers import RobertaForMultipleChoice, RobertaModel, T5EncoderModel
from src.modules.embeddings import RobertaEmbedding, T5Embedding

from src.modules.poolers import T5Pooler


class BaseT5EncoderModule(nn.Module):

    def __init__(self, config: Any) -> None:
        super(BaseT5EncoderModule, self).__init__()
        self.config = config
        # Should we freeze the T5?
        self.embedding = T5Embedding(config)
        self.encoder = T5EncoderModel.from_pretrained("t5-small")

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.pooler = T5Pooler(config)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(sequences)
        encoder_outputs = self.encoder(inputs_embeds=embedding)
        pooler_outputs = self.pooler(encoder_outputs.last_hidden_state)
        return pooler_outputs


class BaseRobertaEncoderModule(nn.Module):

    def __init__(self, config: Any) -> None:
        super(BaseRobertaEncoderModule, self).__init__()
        self.config = config
        # self.embedding = RobertaEmbedding(config)
        self.encoder = RobertaModel.from_pretrained("roberta-base")

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.pooler = T5Pooler(config)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        # embedding = self.embedding(sequences)
        # encoder_outputs = self.encoder(inputs_embeds=embedding)
        encoder_outputs = self.encoder(sequences)
        pooler_outputs = self.pooler(encoder_outputs.last_hidden_state)
        return pooler_outputs
