import os
import torch
import importlib

from typing import Any, Optional
from transformers import PreTrainedTokenizer

from src.runner import Runner


class Trainer:
    model: torch.nn.Module
    save_dir: str
    device: torch.device
    optimizer: torch.optim.Optimizer
    train_runner: Runner
    val_runner: Runner

    def __init__(self,
                 model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 device: torch.device,
                 config: Any,
                 checkpoint: Optional[dict]
                 ) -> None:
        self.model = model
        self.save_dir = config.save_dir
        self.device = device

        # Optimizer
        self.optimizer = self._get_optimizer(config.optimizer)

        if checkpoint is not None:
            print("INFO: Loaded checkpoint. Epoch:",
                  checkpoint["epoch"], "Loss:", checkpoint["loss"])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # DataLoaders
        # TODO: Split dataset, so pre-slice in chunks of context+answer+prediction lines
        create_dataloader = getattr(importlib.import_module(
            f"src.datasets.{config.dataset.name}"), "create_dataloader")
        train_dataloader = create_dataloader(
            tokenizer, config.batch_size)

        # Runners
        self.train_runner = Runner(
            self.model, train_dataloader, device, self.optimizer)
        self.val_runner = Runner(self.model, train_dataloader, device)

    def _get_optimizer(self, config: Any) -> torch.optim.Optimizer:
        if config.type == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=config.params.lr,
                betas=(config.params.beta, 0.999))
        elif config.type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=config.params.lr)
        else:
            raise Exception("Optimizer not set")

        return optimizer

    def _save_train_checkpoint(self, epoch, loss) -> None:
        print("\nNEW BEST MODEL, saving checkpoint.")
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(
            self.save_dir, f"CHECKPOINT_EPOCH_{epoch + 1}.pt"))

    def run_epoch(self) -> None:
        # TODO: Tracker
        print("- TRAINING EPOCH -\n")
        self.train_runner.run_epoch()
        print("- VALIDATION EPOCH -\n")
        self.val_runner.run_epoch()

    def train(self, num_epochs: int) -> None:
        best_val_loss = torch.inf

        for epoch in range(num_epochs):
            print(f'\n\n ---- RUNNING EPOCH {epoch + 1}/{num_epochs} ----\n')
            self.run_epoch()

            train_loss = self.train_runner.average_loss
            train_acc = self.train_runner.average_accuracy

            val_loss = self.val_runner.average_loss
            val_acc = self.val_runner.average_accuracy

            summary = "\t".join([
                f"EPOCH {epoch+1}/{num_epochs}",
                f"train loss {train_loss}",
                f"train acc {train_acc}",
                f"val loss {val_loss}",
                f"val acc {val_acc}"
            ])
            print("\n" + summary + "\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_train_checkpoint(epoch, best_val_loss)

            self.train_runner.reset()
            self.val_runner.reset()
