import torch
import pandas as pd
import numpy as np

from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


DATASET_PATH = "datasets/COMICS/text_cloze_dev_easy.csv"


class ComicsOcrOnlyDataset(Dataset[Any]):
    samples: pd.DataFrame
    tokenizer: PreTrainedTokenizer

    def __init__(self,
                 samples: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 config: Any
                 ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        # We don't want to get the last one
        return 100  # len(self.data) - 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples.iloc(idx)
        context = [
            dialogue["text"]
            for panel in sample["context_panels"]
            for dialogue in panel["dialogues"][:self.config.context_lines]
        ]

        # Be aware that we are padding after concatenating all
        # panel dialogues.
        while len(context) < self.config.context_lines:
            context.insert(0, "")

        context = self.tokenizer(context, return_tensors="pt", truncation=True,
                                 max_length=100, padding="max_length").input_ids
        context = context.view(-1)
        answer_indices = [idx+1]
        indices = torch.randperm(len(self))

        i = 0
        for index in indices:
            if index not in range(idx-self.config.context_lines+1, idx+2):
                answer_indices.append(index)
                i += 1

            if i >= self.config.answer_lines - 1:
                break

        answer_indices = np.array(answer_indices)
        np.random.shuffle(answer_indices)
        answers = self.data[answer_indices].tolist()
        answers = self.tokenizer(answers, return_tensors="pt", truncation=True,
                                 max_length=8, padding="max_length").input_ids
        answers = answers.view(-1)

        targets = torch.tensor(answer_indices == idx+1, dtype=torch.float)

        return {
            "context": context,
            "answers": answers,
            "targets": targets
        }


def create_dataloader(
    tokenizer: PreTrainedTokenizer,
    config: Any
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    # with open(DATASET_PATH, 'r') as f:
    #     data = json.load(f)

    # samples = data["samples"]

    df = pd.read_csv(DATASET_PATH, ',')
    df = df.dropna()

    dataset = ComicsOcrOnlyDataset(df, tokenizer, config.dataset)
    data_len = len(dataset)
    train_len = int(data_len * 0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_len, data_len - train_len])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    return train_dataloader, val_dataloader
