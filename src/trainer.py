import os
import torch
import numpy as np

from typing import Any
from enum import Enum, auto
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src.metrics import Metric


class Stage(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class Trainer:
    model: torch.nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    save_dir: str
    optimizer: torch.optim.Optimizer
    device: torch.device

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader[Any],
                 val_loader: DataLoader[Any],
                 optimizer: torch.optim.Optimizer,
                 config: Any,
                 ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.save_dir = config.save_dir

        # Metrics
        # TODO: When needed, another Metric class will be required
        # Proposal: Create 2 or subclasses that inherit the Metric class
        self.accuracy_metric = Metric()

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'TRAINER --- Selected device: {self.device}.')

        self.model.to(self.device)

    def _reset(self):
        self.accuracy_metric = Metric()

    def _save_train_checkpoint(self, epoch, loss) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(
            self.save_dir, f"CHECKPOINT_EPOCH_{epoch + 1}.pt"))

    def run_epoch(self, stage: Stage) -> torch.Tensor:
        self.model.train(stage is Stage.TRAIN)
        train_loss = []

        # TODO: val_loader ? Maybe it would be ok to create a subclass or sth
        # Proposal: Create a Runner class and use the instances in the Trainer class.
        for local_batch in tqdm(self.train_loader):
            batch = {k: v.to(self.device) for k, v in local_batch.items()}
            outputs = self.model(**batch)

            # Compute Batch Validation Metrics
            targets_np = np.argmax(
                batch["targets"].detach().cpu().numpy(), axis=1)
            outputs_prediction_np = np.argmax(
                outputs.logits.detach().cpu().numpy(), axis=1)
            batch_accuracy: float = accuracy_score(
                targets_np, outputs_prediction_np)
            self.accuracy_metric.update(batch_accuracy, len(batch))

            loss = outputs.loss
            loss.backward()

            if stage is Stage.TRAIN:
                self.optimizer.zero_grad()
                self.optimizer.step()
                # lr_scheduler.step()

            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def train(self, num_epochs: int) -> None:
        best_val_loss = torch.inf

        for epoch in range(num_epochs):
            print(
                '\n\n -------- RUNNING EPOCH {}/{} --------\n'.format(epoch + 1, num_epochs))
            train_loss = self.run_epoch(Stage.TRAIN)
            train_acc = self.accuracy_metric.average
            self._reset()

            if self.val_loader is not None:
                val_loss = self.run_epoch(Stage.VAL)
                val_acc = self.accuracy_metric.average
                self._reset()
            else:
                val_loss = train_loss
                val_acc = train_acc

            # TODO: Make this logging code cleaner
            print('\n EPOCH {}/{} \t train loss {} \t train acc {} \t val loss {} \t val acc {}'.format(epoch +
                                                                                          1, num_epochs, train_loss, train_acc, val_loss, val_acc))

            if val_loss < best_val_loss:
                print("\nNEW BEST MODEL, saving checkpoint.")
                best_val_loss = val_loss
                self._save_train_checkpoint(epoch, best_val_loss)
