"""
Script from https://github.com/ArjanCodes/2021-data-science-refactor/blob/main/after/ds/tensorboard.py
"""
import numpy as np

from pathlib import Path
from typing import List, Tuple
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from src.trackers.tracker import Stage
from src.common.utils import create_experiment_dir


class TensorboardExperiment:
    def __init__(self, log_path: str, experiment_uuid: str = "", create: bool = True):
        log_dir = create_experiment_dir(
            root=log_path, experiment_uuid=experiment_uuid)
        self.stage = Stage.TRAIN
        self._validate_log_dir(log_dir, create=create)
        self._writer = SummaryWriter(log_dir=log_dir)
        plt.ioff()

    def set_stage(self, stage: Stage):
        self.stage = stage

    def flush(self):
        self._writer.flush()

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        elif not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def add_batch_metric(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_confusion_matrix(
        self, y_true: List[np.ndarray], y_pred: List[np.ndarray], step: int
    ):
        y_true, y_pred = self.collapse_batches(y_true, y_pred)
        fig = self.create_confusion_matrix(y_true, y_pred, step)
        tag = f"{self.stage.name}/epoch/confusion_matrix"
        self._writer.add_figure(tag, fig, step)

    @staticmethod
    def collapse_batches(
        y_true: List[np.ndarray], y_pred: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return np.concatenate(y_true), np.concatenate(y_pred)

    def create_confusion_matrix(
        self, y_true: List[np.ndarray], y_pred: List[np.ndarray], step: int
    ) -> plt.Figure:
        cm = ConfusionMatrixDisplay(confusion_matrix(
            y_true, y_pred)).plot(cmap="Blues")
        cm.ax_.set_title(f"{self.stage.name} Epoch: {step}")
        return cm.figure_
