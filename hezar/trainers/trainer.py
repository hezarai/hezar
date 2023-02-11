from typing import Union, List, Dict, Optional

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from hezar.configs import TrainConfig
from hezar.models import Model
from hezar.registry import build_optimizer, build_scheduler


class Trainer:
    def __int__(self,
                model: Union[nn.Module, Model] = None,
                config: TrainConfig = None,
                train_dataset: Optional[Dataset] = None,
                eval_dataset: Optional[Dataset] = None,
                data_collator=None,
                optimizer: optim.Optimizer = None,
                lr_scheduler=None
                ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.train_dataloader, self.eval_dataloader = self._setup_dataloaders()
        self.optimizer, self.lr_scheduler = self._setup_optimizers(optimizer, lr_scheduler)

    def _setup_dataloaders(self):
        train_dataloader = DataLoader(dataset=self.train_dataset,
                                      batch_size=self.config.batch_size,
                                      collate_fn=self.data_collator)
        eval_dataloader = DataLoader(dataset=self.eval_dataset,
                                     batch_size=self.config.batch_size,
                                     collate_fn=self.data_collator)
        return train_dataloader, eval_dataloader

    def _setup_optimizers(self, optimizer=None, lr_scheduler=None):
        optimizer_config = self.config.optimizer.dict()
        optimizer_name = optimizer_config.pop('name')
        scheduler_name = optimizer_name.pop('scheduler')
        if optimizer is None:
            optimizer = build_optimizer(optimizer_name, self.model.parameters(), optimizer_config)
        if lr_scheduler is None:
            lr_scheduler = build_scheduler(scheduler_name, optimizer, optimizer_config.scheduler)
        return optimizer, lr_scheduler

    def train_one_batch(self, batch):
        ...

    def eval_one_batch(self, batch):
        ...

    def train(self):
        ...

    def evaluate(self):
        ...

    def save(self, path):
        ...

    def push_to_hub(self):
        ...

