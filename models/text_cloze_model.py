import torch
from torch import nn
from transformers.modeling_outputs import MultipleChoiceModelOutput

from .base_model import BaseT5EncoderModel


class TextClozeTextOnlyModel(nn.Module):

    def __init__(self, config: object) -> None:
        super(TextClozeTextOnlyModel, self).__init__()
        self.num_labels = config.answer_size
        self.answers_embedding = nn.Embedding(
            config.num_tokens, config.answer_embed_size)
        self.t5_encoder = BaseT5EncoderModel(config)
        self.scores_fc = nn.Linear(
            config.answer_embed_size * config.answer_size *
            config.answer_max_tokens + config.pooler_size,
            config.answer_size
        )
        self.scores_fc_activation = nn.LeakyReLU()

    def forward(self, token_ids, answers, targets):
        outputs = self.t5_encoder(token_ids, answers)
        answer_embedding_outputs = self.answers_embedding(
            answers.view(answers.size(0), -1))
        outputs = torch.cat(
            (outputs, answer_embedding_outputs.view(outputs.size(0), -1)), 1)
        logits = self.scores_fc(outputs)
        loss = None

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, targets)

        # TODO: Hidden states
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
