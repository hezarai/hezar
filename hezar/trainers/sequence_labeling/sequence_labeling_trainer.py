from typing import Callable

import numpy as np
import torch

from ...configs import TrainerConfig
from ...constants import MetricType
from ...data.datasets import Dataset
from ...models import Model
from ...utils import Logger
from ..trainer import Trainer


logger = Logger(__name__)


class SequenceLabelingTrainer(Trainer):
    """
    A trainer for all text classification models

    Args:
        model ([`Model`] or `torch.nn.Module`): The main model to train and evaluate
        config (TrainerConfig): Training configuration and parameters
        train_dataset (Dataset): Train dataset
        eval_dataset (Dataset): Evaluation dataset
        data_collator: Collate function, usually included in the dataset object itself
        optimizer (optim.Optimizer): Model optimizer
        lr_scheduler: Optional learning-rate scheduler

    """
    AVAILABLE_METRICS = [MetricType.SEQEVAL]

    def __init__(
        self,
        model: Model = None,
        config: TrainerConfig = None,
        train_dataset: Dataset = None,
        eval_dataset: Dataset = None,
        data_collator: Callable = None,
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
        self.criterion = torch.nn.CrossEntropyLoss()

    def compute_loss(self, logits, labels, **kwargs) -> torch.Tensor:
        loss = self.criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return loss

    def compute_metrics(self, predictions, labels, **kwargs):
        predictions = np.array(predictions).argmax(2).squeeze()
        labels = np.array(labels).squeeze()

        # Remove ignored index (special tokens) and append `B-` in the beginning for seqeval
        prefix = "" if self.train_dataset.config.is_iob_schema else "B-"
        true_predictions = [
            [f"{prefix}{self.model.config.id2label[p]}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [f"{prefix}{self.model.config.id2label[l]}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = {}
        for metric_name, metric in self.metrics.items():
            x = metric.compute(true_predictions, true_labels)
            results.update(x)
        return results
