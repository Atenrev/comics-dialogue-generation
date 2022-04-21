import torch
import numpy as np

from typing import Any, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.common.registry import Registry

from src.metrics import LossMetric, build_metrics
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
        self.loss_metric = LossMetric()
        self.metrics = build_metrics(Registry.get("model_config").metrics)

    @property
    def average_loss(self) -> float:
        return self.loss_metric.average

    def run_epoch(self, tracker: ExperimentTracker = None) -> None:
        self.model.train(self.stage is Stage.TRAIN)            

        for local_batch in tqdm(self.data_loader):
            batch = {
                k: (v.to(device) 
                if type(v) is torch.Tensor
                else {k2: v2.to(device) for k2, v2 in v.items()}
                if type(v) is dict
                else v)
                for k, v in local_batch.items()
            }
            batch_len = len(batch)
            outputs = self.model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            predictions = np.argmax(logits, axis=1)
            targets = np.argmax(
                batch["targets"].detach().cpu().numpy(), axis=1)
            loss = outputs.loss.detach().cpu().mean().numpy()

            # Compute Batch Metrics
            self.loss_metric.update(loss)

            if tracker is not None:
                tracker.add_batch_metric("loss", loss, self.run_count)
            
            for metric in self.metrics:
                val = metric.calculate_and_update(targets, predictions)     

                if tracker is not None:       
                    tracker.add_batch_metric(metric.name, val, self.run_count)
            

            if self.stage is Stage.TRAIN:
                self.optimizer.zero_grad()
                outputs.loss.mean().backward()
                self.optimizer.step()
                # lr_scheduler.step()

            self.run_count += 1

    def reset(self) -> None:
        self.loss_metric = LossMetric()
        self.metrics = build_metrics(Registry.get("model_config").metrics)
