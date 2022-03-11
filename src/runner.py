import torch
import numpy as np

from typing import Any, Optional
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src.metrics import Metric
from src.trackers.tracker import ExperimentTracker, Stage


class Runner:
    model: torch.nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    data_loader: DataLoader
    device: torch.device
    stage: Stage
    run_count: int

    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader[Any],
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        self.run_count = 0
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device
        self.stage = Stage.TRAIN if optimizer is not None else Stage.VAL

        # Metrics
        # TODO: When needed, another Metric class will be required
        # Proposal: Create 2 or subclasses that inherit the Metric class
        self.loss_metric = Metric()
        self.accuracy_metric = Metric()

    @property
    def average_loss(self) -> float:
        return self.loss_metric.average

    @property
    def average_accuracy(self) -> float:
        return self.accuracy_metric.average

    def run_epoch(self, tracker: ExperimentTracker = None) -> None:
        self.model.train(self.stage is Stage.TRAIN)

        for local_batch in tqdm(self.data_loader):
            batch = {k: v.to(self.device) for k, v in local_batch.items()}
            batch_len = len(batch)
            outputs = self.model(**batch)
            loss = outputs.loss.detach().cpu().numpy()

            # Compute Batch Metrics
            self.loss_metric.update(loss)

            targets_np = np.argmax(
                batch["targets"].detach().cpu().numpy(), axis=1)
            outputs_prediction_np = np.argmax(
                outputs.logits.detach().cpu().numpy(), axis=1)
            batch_accuracy: float = accuracy_score(
                targets_np, outputs_prediction_np)
            self.accuracy_metric.update(batch_accuracy, batch_len)

            if tracker is not None:
                tracker.add_batch_metric("loss", loss, self.run_count)
                tracker.add_batch_metric(
                    "accuracy", batch_accuracy, self.run_count)

            if self.stage is Stage.TRAIN:
                self.optimizer.zero_grad()
                outputs.loss.backward()
                self.optimizer.step()
                # lr_scheduler.step()

            self.run_count += 1

    def reset(self) -> None:
        self.loss_metric = Metric()
        self.accuracy_metric = Metric()
