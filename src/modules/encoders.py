import torch
import math
from typing import Any
from torch import nn
from transformers import RobertaForMultipleChoice, RobertaModel, T5EncoderModel

from src.modules.embeddings import PositionalEncoding, T5Embedding
from src.modules.poolers import BasePooler


class BaseT5EncoderModule(nn.Module):

    def __init__(self, config: Any) -> None:
        super(BaseT5EncoderModule, self).__init__()
        self.config = config
        self.encoder = T5EncoderModel.from_pretrained("t5-small")
        # TODO: Load embedding dynamically
        self.embedding = self.encoder.shared
        self.positional_encoder = PositionalEncoding(
            dim_model=config.embedding_size, 
            dropout_p=config.dropout_p, 
            max_len=512
        )
        self.pooler = BasePooler(config)

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(sequences)
        embedding = embedding * math.sqrt(self.config.embedding_size)
        embedding = self.positional_encoder(embedding)
        encoder_outputs = self.encoder(inputs_embeds=embedding)
        pooler_outputs = self.pooler(encoder_outputs.last_hidden_state)
        return pooler_outputs


class T5HierarchyEncoderModule(BaseT5EncoderModule):

    def __init__(self, config: Any) -> None:
        super(T5HierarchyEncoderModule, self).__init__(config)
        self.embedding = T5Embedding(config)

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(sequences)
        embedding = embedding * math.sqrt(self.config.embedding_size)
        embedding = self.positional_encoder(embedding)
        encoder_outputs = self.encoder(inputs_embeds=embedding)
        pooler_outputs = self.pooler(encoder_outputs.last_hidden_state)
        return pooler_outputs


class BaseRobertaEncoderModule(nn.Module):

    def __init__(self, config: Any) -> None:
        super(BaseRobertaEncoderModule, self).__init__()
        self.config = config
        self.encoder = RobertaModel.from_pretrained("roberta-base")

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.pooler = BasePooler(config)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(sequences)
        pooler_outputs = self.pooler(encoder_outputs.last_hidden_state)
        return pooler_outputs
