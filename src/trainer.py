import os
import logging
import torch
import importlib
import numpy as np

from typing import Any, Optional
from torch.utils.data import DataLoader
from transformers.optimization import Adafactor

from src.common.registry import Registry
from src.runner import Runner
from src.trackers.tensorboard_tracker import TensorboardExperiment
from src.trackers.tracker import ExperimentTracker, Stage


class Trainer:

    def __init__(self,
                 model: torch.nn.Module,
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
            logging.info(f"Loaded checkpoint. Epoch: {checkpoint['epoch']}")
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

        for metric in self.train_runner.metrics:
            self.tracker.add_epoch_metric(
                metric.name, metric.average, epoch_id)

        print("\nVALIDATION EPOCH:\n")
        self.tracker.set_stage(Stage.VAL)
        with torch.no_grad():
            self.val_runner.run_epoch(self.tracker)
        self.tracker.add_epoch_metric(
            "loss", self.val_runner.average_loss, epoch_id)

        for metric in self.val_runner.metrics:
            self.tracker.add_epoch_metric(
                metric.name, metric.average, epoch_id)

    def eval(self, test_dataloader: DataLoader, folds: int = 10) -> None:
        self.test_runner = Runner(self.model, test_dataloader, self.device)

        logging.info(f"Evaluating model. Running {folds} folds on test dataset.")
        final_metrics = {
            "loss": [],
        }

        for metric_name in Registry.get("model_config").metrics:
            final_metrics[metric_name] = []

        for i in range(folds):
            logging.info(f"Running fold {i}...")

            with torch.no_grad():
                self.test_runner.run_epoch()

            final_metrics["loss"].append(self.test_runner.average_loss)

            for metric in self.test_runner.metrics:
                final_metrics[metric.name].append(metric.average)

            self.test_runner.reset()

        report = "Results:\n"

        for metric in final_metrics:
            report += "{}: {:.4f} (±{:.4f})\n".format(
                metric.capitalize(),
                np.mean(final_metrics[metric])*100,
                np.std(final_metrics[metric])*100
            )

        with open(
                os.path.join(
                    self.config.report_path, "eval_report.txt"),
                "w", encoding="utf-8") as f:
            f.write(report)

        print(report)

    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              num_epochs: int) -> None:
        self.train_runner = Runner(
            self.model, train_dataloader, self.device, self.optimizer)
        self.val_runner = Runner(self.model, val_dataloader, self.device)

        self.tracker = TensorboardExperiment(log_path=self.config.runs_path)

        best_val_value = 0

        for epoch in range(num_epochs):
            print(f'\n\n ---- RUNNING EPOCH {epoch + 1}/{num_epochs} ----\n')
            self.run_epoch(epoch)

            train_loss = self.train_runner.average_loss
            train_summary_metrics = "\t".join([
                f"train {metric.name}: {metric.average:.4f}"
                for metric in self.train_runner.metrics
            ])

            val_loss = self.val_runner.average_loss
            val_summary_metrics = "\t".join([
                f"val {metric.name}: {metric.average:.4f}"
                for metric in self.val_runner.metrics
            ])

            summary = "\t".join([
                f"EPOCH {epoch+1}/{num_epochs}",
                f"train loss {train_loss}",
                train_summary_metrics,
                f"val loss {val_loss}",
                val_summary_metrics
            ])
            print("\n" + summary + "\n")

            if val_loss > best_val_value:
                best_val_value = val_loss
                self.tracker.save_checkpoint(epoch, self.model, self.optimizer)

            self.train_runner.reset()
            self.val_runner.reset()
            self.tracker.flush()
