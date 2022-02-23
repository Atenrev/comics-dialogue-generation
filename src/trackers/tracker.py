"""
Script from https://github.com/ArjanCodes/2021-data-science-refactor/blob/main/after/ds/tracking.py
"""
from enum import Enum, auto
from typing import List, Protocol
import numpy as np


class Stage(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class ExperimentTracker(Protocol):
    def set_stage(self, stage: Stage):
        """Sets the current stage of the experiment."""

    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging a epoch-level metric."""

    def add_epoch_confusion_matrix(
        self, y_true: List[np.ndarray], y_pred: List[np.ndarray], step: int
    ):
        """Implements logging a confusion matrix at epoch-level."""