import argparse
import torch
import importlib


from typing import Any
from transformers import T5Tokenizer

from src.configuration import get_configuration
from src.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str,
                        default="configs/text_cloze_text_only.yaml", help='YAML model config')
    parser.add_argument('--model', type=str,
                        default="text_cloze_model", help='Model to run')

    args = parser.parse_args()
    return args


def get_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    if config.type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.params.lr,
            betas=(config.params.beta, 0.999))
    elif config.type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.params.lr)
    else:
        raise Exception("Optimizer not set")

    return optimizer


def main(args) -> None:
    """
    T5-small output size = 512
    """

    # Configuration and checkpoint loading
    config = get_configuration(args.yaml_config)
    # TODO: Checkpoint loader
    checkpoint = None

    # Tokenizer, model and optimizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    ModelClass = getattr(importlib.import_module(
        f"src.models.{args.model}"), config.model.classname)

    model = ModelClass(config.model)
    optimizer = get_optimizer(model, config.optimizer)

    if checkpoint is not None:
        print("INFO: Loaded checkpoint. Epoch:",
              checkpoint["epoch"], "Loss:", checkpoint["loss"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # DataLoaders
    create_dataloader = getattr(importlib.import_module(
        f"src.datasets.{config.dataset.name}"), "create_dataloader")
    train_dataloader = create_dataloader(tokenizer, config.trainer.batch_size)
    # TODO: Split dataset, so pre-slice in chunks of context+answer+prediction lines 

    # Trainer
    trainer = Trainer(model, train_dataloader, None,
                      optimizer, config.trainer)

    trainer.train(config.trainer.epochs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
