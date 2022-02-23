import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


DATASET_PATH = "datasets/COMICS/COMICS_ocr_file.csv"


class ComicsOcrOnlyDataset(Dataset[Any]):
    data: np.ndarray
    tokenizer: PreTrainedTokenizer

    def __init__(self, data, tokenizer: PreTrainedTokenizer, config: Any):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        # We don't want to get the last one
        return 10 # len(self.data) - 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        context = self.data[max(
            0, idx-self.config.context_lines+1):idx+1].tolist()

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
    data = pd.read_csv(DATASET_PATH, ',')
    data = data.dropna()
    data = data["text"].to_numpy()

    dataset = ComicsOcrOnlyDataset(data, tokenizer, config.dataset)
    data_len = len(dataset)
    train_len = int(data_len * 0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_len, data_len - train_len])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_dataloader, val_dataloader
