import os.path
from typing import Union, List, Dict, Optional

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from huggingface_hub import upload_folder

from ..configs import TrainConfig
from ..models import Model
from ..constants import DEFAULT_TRAINER_SUBFOLDER
from ..builders import build_optimizer, build_scheduler
from ..data.datasets import Dataset
from ..utils import resolve_hub_path, get_local_cache_path
from .train_utils import AverageMeter, write_to_tensorboard


class Trainer:
    """
    A general but fully featured model trainer/evaluator in Hezar. Heavily inspired by :class:`transformers.Trainer()`

    Args:
        model ([`Model`] or `torch.nn.Module`): model to train and evaluate
        config (TrainConfig): Training configuration and parameters
        train_dataset (Dataset): Train dataset
        eval_dataset (Dataset): Evaluation dataset
        data_collator: Collate function, usually included in the dataset object itself
        optimizer (optim.Optimizer): Model optimizer
        lr_scheduler: Optional scheduler

    """
    trainer_subfolder = DEFAULT_TRAINER_SUBFOLDER

    def __init__(
            self,
            model: Union[nn.Module, Model] = None,
            config: TrainConfig = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            data_collator=None,
            optimizer: optim.Optimizer = None,
            lr_scheduler=None,
    ):
        self.config = config
        self.num_train_epochs = self.config.num_train_epochs
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.train_dataloader, self.eval_dataloader = self._setup_dataloaders()
        self.optimizer, self.lr_scheduler = self._setup_optimizers(optimizer, lr_scheduler)
        self.loss_tracker = AverageMeter('loss')
        self.tensorboard = SummaryWriter()

    def _setup_dataloaders(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.data_collator,
        )
        eval_dataloader = DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.data_collator,
        )
        return train_dataloader, eval_dataloader

    def _setup_optimizers(self, optimizer=None, lr_scheduler=None):
        if optimizer is None:
            optimizer_config = self.config.optimizer.dict()
            optimizer_name = optimizer_config.pop("name")
            optimizer = build_optimizer(
                optimizer_name,
                self.model.parameters(),
                **optimizer_config,
            )
            if lr_scheduler is None:
                scheduler_name = optimizer_name.pop("scheduler")
                lr_scheduler = build_scheduler(
                    scheduler_name,
                    optimizer,
                    **optimizer_config.scheduler,
                )
        return optimizer, lr_scheduler

    def train_one_batch(self, input_batch):
        """
        Train one batch of data

        Args:
            input_batch: A batch of inputs to train

        Returns:
            The loss value
        """
        outputs = self.model(input_batch)
        if "loss" not in outputs:
            raise ValueError(f"Model outputs must contain `loss`!")
        loss: Tensor = outputs['loss']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate_one_batch(self, input_batch):
        """
        Evaluate one batch of data

        Args:
            input_batch: A batch of inputs to train

        Returns:
            The loss value
        """
        outputs = self.model(input_batch)
        if "loss" not in outputs:
            raise ValueError(f"Model outputs must contain `loss`!")
        loss: Tensor = outputs['loss']

        return loss.item()

    def _one_train_loop(self, epoch_num):
        """
        Train the model for one epoch on the whole train dataset

        Args:
            epoch_num: number of the current epoch

        Returns:
            The average loss through the full iteration
        """
        self.model.train()
        self.loss_tracker.reset()
        with tqdm(self.train_dataloader, unit="batch", desc=f'Epoch: {epoch_num}/{self.num_train_epochs} ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            for input_batch in iterator:
                loss = self.train_one_batch(input_batch)
                self.loss_tracker.update(loss)
                avg_loss = self.loss_tracker.avg
                iterator.set_postfix({"loss": avg_loss})

        return avg_loss

    def evaluate(self):
        self.model.eval()
        with tqdm(self.eval_dataloader, unit="batch", desc=f'Evaluating... ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            with torch.inference_mode():
                for input_batch in iterator:
                    loss = self.evaluate_one_batch(input_batch)
                    self.loss_tracker.update(loss)
                    avg_loss = self.loss_tracker.avg
                    iterator.set_postfix({"loss": avg_loss})
        return avg_loss

    def train(self):
        """
        Train, evaluate, log and save model checkpoints
        """
        for epoch in range(0, self.num_train_epochs + 1):
            print()
            train_loss = self._one_train_loop(epoch)
            eval_loss = self.evaluate()
            self.lr_scheduler.step(eval_loss)

            # tensorboard
            write_to_tensorboard(self.tensorboard, train_loss, 'train', epoch)
            write_to_tensorboard(self.tensorboard, eval_loss, 'val', epoch)

            # save checkpoint
            ckpt_save_path = os.path.join(self.config.checkpoints_dir, str(epoch))
            self.save(ckpt_save_path)

    def save(self, path):
        self.config.save(os.path.join(path, self.trainer_subfolder), filename='train_config.yaml')
        self.model.save(path, save_config=True)
        self.train_dataset.preprocessor.save(path)

    def push_to_hub(self, hub_path, commit_message=None):
        """
        Push everything to the Hub

        Args:
            hub_path: Path to hub
            commit_message: Commit message for the push
        """
        hub_path = resolve_hub_path(hub_path)
        cache_path = get_local_cache_path(hub_path, repo_type="model")

        self.save(cache_path)

        if not commit_message:
            commit_message = f"Hezar: Upload with Trainer"

        upload_folder(
            repo_id=hub_path,
            folder_path=cache_path,
            repo_type="model",
            commit_message=commit_message,
        )



