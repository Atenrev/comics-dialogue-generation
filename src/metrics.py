import torch
import sacrebleu
import numpy as np

from typing import Any, List
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score


@dataclass
class Metric:
    """
    Taken from https://github.com/ArjanCodes/2021-data-science-refactor/blob/main/after/ds/metrics.py
    """
    name: str = field(init=False)
    inpyt_type: str = field(init=False)
    values: List[float] = field(default_factory=list)
    running_total: float = 0.0
    num_updates: float = 0.0
    average: float = 0.0

    def update(self, value: float, batch_size: int = 1) -> None:
        self.values.append(value)
        self.running_total += value * batch_size
        self.num_updates += batch_size
        self.average = self.running_total / self.num_updates

    def calculate_and_update(self, targets: Any, predictions: Any) -> float:
        raise NotImplementedError


class LossMetric(Metric):
    name: str = "loss"
    inpyt_type: str = "float"
    pass


class AccuracyMetric(Metric):
    name: str = "accuracy"
    inpyt_type: str = "np.ndarray"

    def calculate_and_update(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        batch_len = len(targets)
        batch_accuracy = accuracy_score(targets, predictions)
        batch_accuracy = np.mean(batch_accuracy)
        self.update(batch_accuracy, batch_len)
        return batch_accuracy


class BLEUMetric(Metric):
    name: str = "bleu"
    inpyt_type: str = "str"
    
    def calculate_and_update(self, targets: List[List[str]], predictions: List[str]) -> float:
        bleu = sacrebleu.corpus_bleu(predictions, targets)
        self.update(bleu.score, len(targets))
        return bleu.score


def build_metrics(metrics_names: List[str]) -> List[Metric]:
    if metrics_names is None:
        return []

    import sys, inspect
    metrics = [
        obj
        for _, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isclass(obj) and obj is not Metric 
        and obj.name in metrics_names
    ]

    # for metric_name in metrics_names:
    #     if metric_name == "accuracy":
    #         metrics.append(AccuracyMetric())
    #     else:
    #         raise ValueError(f"Unknown metric: {metric_name}")

    return metrics