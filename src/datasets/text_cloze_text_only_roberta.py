import torch
import pandas as pd
import numpy as np

from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


TRAIN_DATASET_PATH = "datasets/COMICS/text_cloze_dev_easy.csv"
VAL_DATASET_PATH = "datasets/COMICS/text_cloze_test_easy.csv"


class ComicsOcrOnlyDataset(Dataset[Any]):
    data: pd.DataFrame
    tokenizer: PreTrainedTokenizer

    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 config: Any
                 ):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data.iloc[idx]
        context = [
            "[C00]" + sample["context_text_0_0"],
            "[C01]" + sample["context_text_0_1"],
            "[C02]" + sample["context_text_0_2"],
            "[C10]" + sample["context_text_1_0"],
            "[C11]" + sample["context_text_1_1"],
            "[C12]" + sample["context_text_1_2"],
            "[C20]" + sample["context_text_2_0"],
            "[C21]" + sample["context_text_2_1"],
            "[C22]" + sample["context_text_2_2"],
        ]

        context = self.tokenizer(context, return_tensors="pt", truncation=True,
                                 max_length=self.config.context_max_speech_size, padding="max_length").input_ids
        context = context.view(-1)
        
        answers = [
            "[A0]" + sample["answer_candidate_0_text"],
            "[A1]" + sample["answer_candidate_1_text"],
            "[A2]" + sample["answer_candidate_2_text"],
        ]

        answers = self.tokenizer(answers, return_tensors="pt", truncation=True,
                                 max_length=self.config.answer_max_tokens, padding="max_length").input_ids
        answers = answers.view(-1)

        return {
            "context": context,
            "answers": answers,
            "targets": sample["correct_answer"]
        }


def create_dataloader(
    tokenizer: PreTrainedTokenizer,
    config: Any
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    # with open(DATASET_PATH, 'r') as f:
    #     data = json.load(f)

    # samples = data["samples"]

    dev_df = pd.read_csv(VAL_DATASET_PATH, ',')
    dev_df = dev_df.fillna("")
    test_df = pd.read_csv(TRAIN_DATASET_PATH, ',')
    test_df = test_df.fillna("")

    df = pd.concat((dev_df, test_df))

    train_dataset = ComicsOcrOnlyDataset(df, tokenizer, config.dataset)
    # val_dataset = ComicsOcrOnlyDataset(val_df, tokenizer, config.dataset)
    data_len = len(train_dataset)
    train_len = int(data_len * 0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_len, data_len - train_len])

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
