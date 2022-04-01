import logging
import torch
import importlib

from typing import Any, Optional
from transformers import PreTrainedTokenizer
from transformers.optimization import Adafactor

from src.runner import Runner
from src.trackers.tensorboard_tracker import TensorboardExperiment
from src.trackers.tracker import ExperimentTracker, Stage


class Trainer:
    experiment_name: str
    config: Any
    tracker: ExperimentTracker
    model: torch.nn.Module
    dataset: Any
    device: torch.device
    optimizer: torch.optim.Optimizer
    train_runner: Runner
    val_runner: Runner

    def __init__(self,
                 model: torch.nn.Module,
                 dataset_dir: str,
                 dataset_config: Any,
                 tokenizer: PreTrainedTokenizer,
                 device: torch.device,
                 config: Any,
                 checkpoint: Optional[dict] = None,
                 ) -> None:
        self.model = model
        self.device = device
        self.config = config

        # Optimizer
        self.optimizer = self._get_optimizer(config.optimizer)

        if checkpoint is not None:
            logging.info(f"Loaded checkpoint. Epoch: {checkpoint['epoch']} Loss: {checkpoint['loss']}")
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # DataLoaders
        create_dataloader = getattr(importlib.import_module(
            f"src.datasets.{dataset_config.name}"), "create_dataloader")
        train_dataloader, val_dataloader, test_dataloader = create_dataloader(
            tokenizer, config.batch_size, dataset_dir, dataset_config)

        # Runners
        self.train_runner = Runner(
            self.model, train_dataloader, device, self.optimizer)
        self.val_runner = Runner(self.model, val_dataloader, device)
        self.test_runner = Runner(self.model, test_dataloader, device)

    def _get_optimizer(self, config: Any) -> torch.optim.Optimizer:
        if config.type == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.params.lr,
                betas=(config.params.beta, 0.999)
            )
        elif config.type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.params.lr
            )
        elif config.type == "adafactor":
            optimizer = Adafactor(
                self.model.parameters(),
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
                clip_threshold=1.0,
                lr=None
            )
        else:
            raise Exception("Optimizer not set")

        return optimizer        

    def run_epoch(self, epoch_id: int) -> None:
        print("\nTRAINING EPOCH:\n")
        self.tracker.set_stage(Stage.TRAIN)
        self.train_runner.run_epoch(self.tracker)
        self.tracker.add_epoch_metric(
            "loss", self.train_runner.average_loss, epoch_id)
        self.tracker.add_epoch_metric(
            "accuracy", self.train_runner.average_accuracy, epoch_id)

        print("\nVALIDATION EPOCH:\n")
        self.tracker.set_stage(Stage.VAL)
        with torch.no_grad():
            self.val_runner.run_epoch(self.tracker)
        self.tracker.add_epoch_metric(
            "loss", self.val_runner.average_loss, epoch_id)
        self.tracker.add_epoch_metric(
            "accuracy", self.val_runner.average_accuracy, epoch_id)

    def eval(self) -> None:
        print("\nTEST EPOCH:\n")
        with torch.no_grad():
            self.test_runner.run_epoch()
        val_loss = self.test_runner.average_loss
        val_acc = self.test_runner.average_accuracy
        summary = "\t".join([
            f"TEST EPOCH",
            f"test loss {val_loss}",
            f"test acc {val_acc}"
        ])
        print("\n" + summary + "\n")

    def train(self, num_epochs: int) -> None:
        self.tracker = TensorboardExperiment(log_path=self.config.runs_path)
        best_val_acc = 0

        for epoch in range(num_epochs):
            print(f'\n\n ---- RUNNING EPOCH {epoch + 1}/{num_epochs} ----\n')
            self.run_epoch(epoch)

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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.tracker.save_checkpoint(epoch, self.model, self.optimizer)

            self.train_runner.reset()
            self.val_runner.reset()
            self.tracker.flush()
