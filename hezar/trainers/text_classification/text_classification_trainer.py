from typing import Callable, Dict, Iterable, Union

import torch

from hezar.constants import MetricType

from ... import Metric, snake_case
from ...builders import build_metric
from ...configs import TrainerConfig
from ...constants import (
    DEFAULT_DATASET_CONFIG_FILE,
    DEFAULT_TRAINER_CONFIG_FILE,
    DEFAULT_TRAINER_SUBFOLDER,
)
from ...data.datasets import Dataset
from ...models import Model
from ...utils import get_logger
from ..trainer import Trainer
from ..trainer_utils import MetricsManager

logger = get_logger(__name__)


class TextClassificationTrainer(Trainer):
    """
    A general but fully featured model trainer/evaluator for Hezar models.

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
            lr_scheduler=lr_scheduler)

    def _setup_metrics_manager(self, metrics: Iterable[Union[str, Callable, Metric, MetricType]]) -> MetricsManager:
        """
        Set up metrics manager to track and update metrics like loss, accuracy, f1, etc.

        Args:
            metrics: A dict of metrics names and their kwargs {metric_name: **kwargs}

        Returns:
             A MetricsManager instance
        """
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

    def training_step(self, input_batch: Dict[str, torch.Tensor]):
        """
        Train one batch of data and return metrics outputs

        Args:
            input_batch: A batch of inputs to train

        Returns:
            The metrics results
        """
        input_batch = {k: v.to(self.device) for k, v in input_batch.items() if isinstance(v, torch.Tensor)}

        with torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype, enabled=self.config.use_amp):
            outputs = self.model(input_batch)
            if "loss" not in outputs:
                raise ValueError("Model outputs must contain `loss`!")
            loss: torch.Tensor = outputs["loss"]

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        results = self.metrics_manager.compute(outputs["logits"].detach().cpu(), input_batch["labels"].detach().cpu())
        results["loss"] = loss.item()

        return results

    def evaluation_step(self, input_batch: Dict[str, torch.Tensor]):
        """
        Evaluate one batch of data and return metrics outputs

        Args:
            input_batch: A batch of inputs to train

        Returns:
            The metrics results
        """
        input_batch = {k: v.to(self.device) for k, v in input_batch.items() if isinstance(v, torch.Tensor)}

        with torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype, enabled=self.config.use_amp):
            outputs = self.model(input_batch)
            if "loss" not in outputs:
                raise ValueError("Model outputs must contain `loss`!")
            loss: torch.Tensor = outputs["loss"]

        results = self.metrics_manager.compute(outputs["logits"].detach().cpu(), input_batch["labels"].detach().cpu())
        results["loss"] = loss.item()

        return results
