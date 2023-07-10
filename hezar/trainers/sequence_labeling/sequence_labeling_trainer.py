from typing import Callable, Dict, Iterable, Union, Tuple

import torch

from hezar.constants import MetricType

from ...builders import build_metric
from ...configs import TrainerConfig
from ...constants import (
    DEFAULT_DATASET_CONFIG_FILE,
    DEFAULT_TRAINER_CONFIG_FILE,
    DEFAULT_TRAINER_SUBFOLDER,
)
from ...data.datasets import Dataset
from ...models import Model
from ...metrics import Metric
from ...utils import get_logger, snake_case
from ..trainer import Trainer
from ..trainer_utils import MetricsManager

logger = get_logger(__name__)


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

    trainer_subfolder = DEFAULT_TRAINER_SUBFOLDER
    trainer_config_file = DEFAULT_TRAINER_CONFIG_FILE
    dataset_config_file = DEFAULT_DATASET_CONFIG_FILE

    def __init__(
        self,
        model: Model = None,
        config: TrainerConfig = None,
        train_dataset: Dataset = None,
        eval_dataset: Dataset = None,
        data_collator=None,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler=None,
    ):
        super().__init__(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def _setup_metrics_manager(self, metrics: Iterable[Union[str, Callable, Metric, MetricType]]) -> MetricsManager:
        metrics_dict = {"loss": None}
        for metric in metrics:
            if isinstance(metric, str):
                metrics_dict[metric] = build_metric(metric)
            elif isinstance(metric, Metric):
                metrics_dict[metric] = metric.compute
            elif callable(metric):
                if hasattr(metric, "compute"):
                    metrics_dict[snake_case(metric.__name__)] = metric.compute
                else:
                    metrics_dict[snake_case(metric.__name__)] = metric
            else:
                raise ValueError(f"Metric {metric} is not a valid metric!")
        metrics_manager = MetricsManager(metrics_dict)
        return metrics_manager

    def compute_loss(self, logits, labels, **kwargs) -> torch.Tensor:
        loss = self.criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return loss
