import torch

from ...configs import TrainerConfig
from ...data.datasets import Dataset
from ...models import Model
from ...utils import get_logger
from ..trainer import Trainer

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
    AVAILABLE_METRICS = []

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

    def compute_loss(self, logits, labels, **kwargs) -> torch.Tensor:
        loss = self.criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return loss

    def compute_metrics(self, predictions, labels, **kwargs):
        raise NotImplementedError
