import os
import logging
import torch
import numpy as np

from typing import Any, Optional, Tuple
from torch.utils.data import DataLoader
from transformers.optimization import Adafactor

from src.common.registry import Registry
from src.common.utils import generate_experiment_name
from src.runner import Runner
from src.trackers.tensorboard_tracker import TensorboardExperiment
from src.trackers.tracker import Stage


class Trainer:

    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: DataLoader[Any],
                 val_dataloader: DataLoader[Any],
                 test_dataloader: DataLoader[Any],
                 device: torch.device,
                 config: Any,
                 checkpoint: Optional[dict] = None,
                 ) -> None:
        """
        Initialize the trainer.

        Args:
            model: The model to train.
            device: The device to train the model on.
            config: The configuration of the Trainer.
            checkpoint: The checkpoint to load the optimizer and model from.
            """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.config = config

        # Optimizer
        self.optimizer, self.scheduler = self._get_optimizer_and_scheduler(
            config.optimizer)

        if checkpoint is not None:
            logging.info(f"Loaded checkpoint. Epoch: {checkpoint['epoch']}")
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def _get_optimizer_and_scheduler(self, config: Any) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LambdaLR]]:
        """
        Get the optimizer for the model.

        Args:
            config: The optimizer configuration.

        Returns:
            The optimizer and scheduler.
        """
        scheduler = None

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
        elif config.type == "adamw":
            from transformers.optimization import AdamW, get_linear_schedule_with_warmup
            batch_per_epoch = len(self.train_dataloader)
            t_total = batch_per_epoch // config.gradient_accumulation_steps * self.config.epochs
            warmup_ratio = config.warmup_ratio
            warmup_iters = int(t_total * warmup_ratio)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=config.lr, eps=config.adam_eps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, warmup_iters, t_total)
        else:
            raise Exception("Optimizer not set")

        return optimizer, scheduler

    def run_epoch(self, epoch_id: int) -> None:
        """
        Run an epoch of training.

        Args:
            epoch_id: The id of the epoch.
        """
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

    def eval(self) -> None:
        """
        Evaluate the model on the given dataloader for the given number of folds.
        Save the results to report_path.
        """
        import json
        self.test_runner = Runner(
            self.model, self.test_dataloader, self.device)

        logging.info("Evaluating model.")

        with torch.no_grad():
            self.test_runner.run_epoch()

        results = {
            "loss": self.test_runner.average_loss,
            "metrics": [],
        }

        for metric in self.test_runner.metrics:
            results["metrics"] += [{
                "name": metric.name,
                "value": metric.average,
            }]

        # predictions = self.test_runner.predictions_info
        # correct_predictions = []
        # incorrect_predictions = []

        # for sample_id in predictions:
        #     prediction = predictions[sample_id]["prediction"]
        #     target = predictions[sample_id]["target"]

        #     if prediction == target:
        #         correct_predictions.append(sample_id)
        #     else:
        #         incorrect_predictions.append(sample_id)

        # results["correct_predictions"] = correct_predictions
        # results["incorrect_predictions"] = incorrect_predictions

        experiment_name = generate_experiment_name()

        with open(os.path.join(
                self.config.report_path, f"{experiment_name}_report.json"),
                "w", encoding="utf-8") as f:
            json.dump(results, f)

    def train(self,
              num_epochs: int) -> None:
        """
        Train the model for num_epochs epochs on the given dataloaders.

        Args:
            num_epochs: The number of epochs to train for.
        """
        self.train_runner = Runner(
            self.model, self.train_dataloader, self.device,
            self.optimizer, self.scheduler)
        self.val_runner = Runner(self.model, self.val_dataloader, self.device)

        experiment_name = generate_experiment_name()
        self.tracker = TensorboardExperiment(log_path=self.config.runs_path,
                                             experiment_name=experiment_name)

        best_val_value = float("inf")

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

            if val_loss < best_val_value:
                best_val_value = val_loss
                self.tracker.save_checkpoint(epoch, self.model, self.optimizer)

            self.train_runner.reset()
            self.val_runner.reset()
            self.tracker.flush()
