import torch
from torch import nn
from transformers import T5EncoderModel


class T5Pooler(nn.Module):
    """
    Based on https://www.kaggle.com/debarshichanda/explore-t5
    """

    def __init__(self, config: object, activation=nn.LeakyReLU()):
        super().__init__()
        self.dense = nn.Linear(config.encoder_size, config.pooler_size)
        self.activation = activation

    def forward(self, hidden_states):
        mean_tensor = torch.mean(hidden_states, dim=1)
        pooled_output = self.dense(mean_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BaseT5EncoderModel(nn.Module):

    def __init__(self, config: object) -> None:
        super(BaseT5EncoderModel, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained("t5-small")
        self.pooler = T5Pooler(config)

    def forward(self, context, answers):
        X = self.encoder(context)
        X = self.pooler(X.last_hidden_state)
        return X
