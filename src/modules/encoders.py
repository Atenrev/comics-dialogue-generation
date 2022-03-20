import torch
import math
from typing import Any
from torch import nn
from transformers import AutoModel, RobertaForMultipleChoice, RobertaModel, T5EncoderModel

from src.modules.embeddings import PositionalEncoding, T5Embedding
from src.modules.poolers import MeanPooler


class BaseT5EncoderModule(nn.Module):

    def __init__(self, config: Any) -> None:
        super(BaseT5EncoderModule, self).__init__()
        self.config = config
        self.encoder = T5EncoderModel.from_pretrained(config.architecture)
        # TODO: Load embedding dynamically
        self.embedding = self.encoder.shared
        self.positional_encoder = PositionalEncoding(
            dim_model=config.embedding_size, 
            dropout_p=config.dropout_p, 
            max_len=config.embedding_size
        )
        self.pooler = MeanPooler(config)

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
        # self.embedding = T5Embedding(config)
        self.embedding = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.fc_embedding = nn.Sequential(
            nn.Linear(384, self.config.embedding_size),
            nn.ReLU()
        )

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        batch_size = sequences.size(0)
        sequences = sequences.view(-1, sequences.size(2))
        embedding = self.embedding(sequences)
        embedding = embedding.pooler_output
        embedding = embedding.view(batch_size, -1, 384)
        embedding = self.fc_embedding(embedding)
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

        self.pooler = MeanPooler(config)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(sequences)
        pooler_outputs = self.pooler(encoder_outputs.last_hidden_state)
        return pooler_outputs
