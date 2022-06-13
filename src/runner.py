import torch
import numpy as np

from typing import Any, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.common.registry import Registry

from src.metrics import LossMetric, build_metrics
from src.models.base_model import BaseModel
from src.trackers.tracker import ExperimentTracker, Stage


class Runner:

    def __init__(
        self,
        model: BaseModel,
        data_loader: DataLoader[Any],
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    ) -> None:
        """
        Runner for training and evaluation.

        Args:
            model: The model to train.
            data_loader: The data loader to use.
            device: The device to train the model on.
            optimizer: The optimizer to use.
        """
        self.run_count = 0
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_loader = data_loader
        self.device = device
        self.stage = Stage.TRAIN if optimizer is not None else Stage.VAL

        # if self.stage is Stage.VAL:
        #     self.predictions_info = {}

        # Metrics
        self.loss_metric = LossMetric()
        self.metrics = build_metrics(Registry.get("model_config").metrics)

    @property
    def average_loss(self) -> float:
        return self.loss_metric.average

    def run_epoch(self, tracker: Optional[ExperimentTracker] = None) -> None:
        """
        Run an epoch of training or evaluation.

        Args:
            tracker: The ExperimentTracker to use.
        """
        self.model.train(self.stage is Stage.TRAIN)

        for local_batch in tqdm(self.data_loader):
            batch = local_batch["data"] if "data" in local_batch else local_batch
            outputs = self.model.run(**batch)
            predictions = outputs.prediction.detach().cpu().numpy()
            targets = batch["target"].detach().cpu().numpy()
            loss = outputs.loss.detach().cpu().mean().numpy()

            # Compute Batch Metrics
            self.loss_metric.update(loss)

            if tracker is not None:
                tracker.add_batch_metric("loss", loss, self.run_count)

            for metric in self.metrics:
                if metric.inpyt_type == "str":
                    target_texts = batch["target_text"]
                    if isinstance(self.model, torch.nn.DataParallel):
                        tokenizer = self.model.module.tokenizer
                    else:
                        tokenizer = self.model.tokenizer
                    predictions_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                    val = metric.calculate_and_update(target_texts, predictions_texts)
                else:
                    val = metric.calculate_and_update(targets, predictions)

                if tracker is not None:
                    tracker.add_batch_metric(metric.name, val, self.run_count)

            # if self.stage is Stage.VAL:
            #     for sample_id, prediction, target in zip(
            #             batch["sample_id"], predictions, targets):
            #         self.predictions_info[sample_id] = {
            #             "prediction": prediction,
            #             "target": target,
            #         }
            
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                outputs.loss.mean().backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

            self.run_count += 1

    def reset(self) -> None:
        """
        Reset the metrics.
        """
        self.loss_metric = LossMetric()
        self.metrics = build_metrics(Registry.get("model_config").metrics)
