from typing import Any
import torch
from torch import nn
from transformers import RobertaModel, T5EncoderModel


class T5Embedding(nn.Module):

    def __init__(self, config: Any) -> None:
        super(T5Embedding, self).__init__()
        self.config = config
        self.embedding = T5EncoderModel.from_pretrained("t5-small")

        # for param in self.embedding.parameters():
        #     param.requires_grad = False

        self.fc1 = nn.Sequential(
            nn.Linear(512, config.embedding_size),
            nn.ReLU()
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        batch_size = sequences.size(0)
        sequence_length = sequences.size(2)
        sequences = sequences.view(-1, sequence_length)
        embedding = self.embedding(sequences).last_hidden_state
        embedding = embedding.view(batch_size, -1, 512)
        embedding = self.fc1(embedding)
        return embedding


class RobertaEmbedding(nn.Module):

    def __init__(self, config: Any) -> None:
        super(RobertaEmbedding, self).__init__()
        self.config = config
        self.embedding = RobertaModel.from_pretrained("roberta-base")

        # for param in self.embedding.parameters():
        #     param.requires_grad = False

        self.fc1 = nn.Sequential(
            nn.Linear(768, config.embedding_size),
            nn.ReLU()
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        batch_size = sequences.size(0)
        sequence_length = sequences.size(2)
        sequences = sequences.view(-1, sequence_length)
        embedding = self.embedding(sequences).last_hidden_state
        embedding = embedding.view(batch_size, -1, 768)
        embedding = self.fc1(embedding)
        return embedding
