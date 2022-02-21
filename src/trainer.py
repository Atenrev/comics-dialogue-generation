import os
import torch

from typing import Any
from enum import Enum, auto
from tqdm import tqdm
from torch.utils.data import DataLoader


class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


class Trainer:
    model: torch.nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    save_dir: str
    lr: float
    optimizer: torch.optim.Optimizer
    device: torch.device

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Any,
                 ) -> None:
        # torch.manual_seed(42)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = config.save_dir

        self.lr = config.learning_rate

        if config.optimizer.name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr,
                betas=(config.optimizer.parameters.beta, 0.999))
        elif config.optimizer.name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr)
        else:
            raise Exception("Optimizer not set")

        # if checkpoint is not None:
        #     print("INFO: Loaded checkpoint. Epoch:",
        #           checkpoint["epoch"], "Loss:", checkpoint["loss"])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     self.init_epoch = checkpoint['epoch']

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'TRAINER --- Selected device: {self.device}.')

        self.model.to(self.device)

    def _save_train_checkpoint(self, epoch, loss) -> None:
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

        for local_batch in tqdm(self.train_loader):
            batch = {k: v.to(self.device) for k, v in local_batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            if stage is Stage.TRAIN:
                self.optimizer.zero_grad()
                self.optimizer.step()
                # lr_scheduler.step()

            train_loss.append(loss.detach().cpu().numpy())

        return torch.mean(torch.tensor(train_loss))

    def train(self, num_epochs: int) -> None:
        best_val_loss = torch.inf

        for epoch in range(num_epochs):
            print(
                '\n\n -------- RUNNING EPOCH {}/{} --------\n'.format(epoch + 1, num_epochs))
            train_loss = self.run_epoch(Stage.TRAIN)

            if self.val_loader is not None:
                val_loss = self.run_epoch(Stage.TEST)
            else:
                val_loss = train_loss
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch +
                                                                          1, num_epochs, train_loss, val_loss))

            if val_loss < best_val_loss:
                # TODO: Save a checkpoint instead of only a path.
                best_val_loss = val_loss
                self._save_train_checkpoint(epoch, best_val_loss)
