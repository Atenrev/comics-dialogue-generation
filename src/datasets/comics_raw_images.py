import torch
import glob
import os

from PIL import Image
from typing import Any, Tuple, List, Optional
from torch.utils.data import DataLoader

from src.sample import Sample
from src.datasets.base_dataset import BaseDataset


class ComicsRawImages(BaseDataset):

    def __init__(self,
                 image_paths: List[str],
                 device: torch.device,
                 config: Any,
                 transform: Any = None,
                 feature_extractor: Any = None
                 ):
        super().__init__(device, config)
        self.image_paths = image_paths
        self.transform = transform
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def getitem(self, idx: int) -> Sample:
        image_path = self.image_paths[idx]

        # Generates the id of the sample from the name of the image and the parent directory.
        sample_id = os.path.basename(
            os.path.dirname(image_path)) + "_" + os.path.basename(image_path)
        sample_id = sample_id.split(".")[0]

        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        elif self.feature_extractor:
            image = self.feature_extractor(image, return_tensors="pt")
            image["pixel_values"] = image["pixel_values"].squeeze(0)
        else:
            raise ValueError("Either transform or feature_extractor must be provided")

        return Sample(sample_id, {
            "image": image,
        })


def create_dataloader(
    batch_size: int,
    dataset_path: str,
    device: torch.device,
    config: Any,
    inference: bool = False,
    dataset_kwargs: dict = {},
) -> Tuple[DataLoader[Any], Optional[DataLoader[Any]], Optional[DataLoader[Any]]]:
    image_paths = glob.glob(os.path.join(dataset_path, "**/*.png"), recursive=True)

    dataset = ComicsRawImages(image_paths, device, config, **dataset_kwargs)

    if not inference:
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
    else:
        val_dataloader = test_dataloader = None
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        

    return train_dataloader, val_dataloader, test_dataloader
