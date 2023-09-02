from typing import List

import numpy as np
import torch

from ...configs import TrainerConfig
from ...constants import MetricType
from ...data.datasets import Dataset
from ...models import Model
from ...utils import Logger
from ..trainer import Trainer


logger = Logger(__name__)


class TextClassificationTrainer(Trainer):
    """
    A trainer for all text classification models

    Args:
        model ([`Model`] or `torch.nn.Module`): The main model to train and evaluate
        config (TrainerConfig): Training configuration and parameters
        train_dataset (Dataset): Train dataset
        eval_dataset (Dataset): Evaluation dataset
        data_collator: Collate function, usually included in the dataset object itself
        preprocessor: Preprocessor object
        optimizer (optim.Optimizer): Model optimizer
        lr_scheduler: Optional learning-rate scheduler

    """
    AVAILABLE_METRICS = [
        MetricType.ACCURACY,
        MetricType.RECALL,
        MetricType.PRECISION,
        MetricType.F1,
    ]

    def __init__(
        self,
        model: Model = None,
        config: TrainerConfig = None,
        train_dataset: Dataset = None,
        eval_dataset: Dataset = None,
        data_collator=None,
        preprocessor=None,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler=None,
    ):
        super().__init__(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            preprocessor=preprocessor,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, logits, labels, **kwargs) -> torch.Tensor:
        loss = self.criterion(logits, labels)
        return loss

    def compute_metrics(self, predictions: List[np.ndarray], labels: List[np.ndarray], **kwargs):
        predictions = np.array(predictions).argmax(1).flatten()
        labels = np.array(labels).flatten()
        results = {}
        for metric_name, metric in self.metrics.items():
            results.update(metric.compute(predictions, labels))
        return results

