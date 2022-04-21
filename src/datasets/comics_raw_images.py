import torch
import glob
import pandas as pd

from PIL import Image
from typing import Any, Dict, Tuple, List
from torch.utils.data import DataLoader, Dataset


class ComicsRawImages(Dataset[Any]):

    def __init__(self,
                 image_paths: List[str],
                 config: Any,
                 transform: Any = None,
                 feature_extractor: Any = None
                 ):
        self.image_paths = image_paths
        self.transform = transform
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        elif self.feature_extractor:
            image = self.feature_extractor(image, return_tensors="pt")
            image["pixel_values"] = image["pixel_values"].squeeze(0)
        else:
            raise ValueError("Either transform or feature_extractor must be provided")

        return {
            "image_id": image_path,
            "image": image,
        }


def create_dataloader(
    batch_size: int,
    dataset_path: str,
    config: Any,
    dataset_kwargs: dict = {},
) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    # Make a list of all the image paths given the dataset_path
    image_paths = glob.glob(f"{dataset_path}/*.png")

    dataset = ComicsRawImages(image_paths, config, **dataset_kwargs)
    # Split the dataset into train, validation, and test
    train_size = int(0.8 * len(image_paths))
    val_size = int(0.1 * len(image_paths))
    test_size = len(image_paths) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    return train_dataloader, val_dataloader, test_dataloader
