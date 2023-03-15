import os
from typing import Dict, List, Tuple
import random

import torch
from huggingface_hub import upload_folder, hf_hub_download
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, F1Score, Precision
from tqdm import tqdm
import numpy as np

from ..builders import build_optimizer, build_scheduler
from ..configs import TrainConfig
from ..constants import (
    DEFAULT_TRAINER_CONFIG_FILE,
    DEFAULT_TRAINER_SUBFOLDER,
    HEZAR_TMP_DIR,
    DEFAULT_DATASET_CONFIG_FILE,
    TQDM_BAR_FORMAT,
)
from ..data.datasets import Dataset
from ..models import Model
from ..utils import get_local_cache_path, get_logger, resolve_pretrained_path
from .trainer_utils import MetricsManager, write_to_tensorboard


logger = get_logger(__name__)

METRICS_MAP = {
    "accuracy": Accuracy,
    "f1": F1Score,
    "precision": Precision,
}


class Trainer:
    """
    A general but fully featured model trainer/evaluator for Hezar models.

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
    trainer_config_file = DEFAULT_TRAINER_CONFIG_FILE
    dataset_config_file = DEFAULT_DATASET_CONFIG_FILE

    def __init__(
        self,
        model: Model = None,
        config: TrainConfig = None,
        train_dataset: Dataset = None,
        eval_dataset: Dataset = None,
        data_collator=None,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler=None,
    ):
        self.config = config
        self.num_train_epochs = self.config.num_train_epochs
        self.device = self.config.device if self.config.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = self._init_model_weights(model).to(self.device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.num_labels = self.train_dataset.num_labels  # noqa
        self.set_seed(self.config.seed)
        self.train_dataloader, self.eval_dataloader = self._setup_dataloaders()
        self.optimizer, self.lr_scheduler = self._setup_optimizers(optimizer, lr_scheduler)
        self.metrics_manager = self._setup_metrics_manager(self.config.metrics)
        self.tensorboard = SummaryWriter()

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _init_model_weights(self, model):
        weights_path = self.config.init_weights_from
        hub_path = resolve_pretrained_path(weights_path)
        local_path = hf_hub_download(hub_path, filename=model.model_filename, cache_dir=HEZAR_TMP_DIR)
        model.load_state_dict(torch.load(local_path))
        return model

    def _setup_dataloaders(self):
        """
        Set up data loaders (train/eval) and return them.

        Returns:
             A tuple of train and eval dataloaders
        """
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
        )
        eval_dataloader = DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
        )
        return train_dataloader, eval_dataloader

    def _setup_optimizers(self, optimizer: torch.optim.Optimizer = None, lr_scheduler=None):
        """
        Set up the optimizer and lr scheduler if they're not already given

        Args:
            optimizer: If None do nothing and return it, otherwise build it using the train config
            lr_scheduler: If None do nothing and return it, otherwise build it using the train config

        Returns:
            Optimizer and scheduler
        """
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

    def _setup_metrics_manager(self, metrics: Dict[str, Dict]) -> MetricsManager:
        """
        Set up metrics manager to track and update metrics like loss, accuracy, f1, etc.

        Args:
            metrics: A list of metrics in tuple format (metric_name, metric_kwargs)

        Returns:
             A MetricsManager instance
        """
        metrics_dict = {"loss": None}
        for name, kwargs in metrics.items():
            metrics_dict[name] = METRICS_MAP[name](num_classes=self.num_labels, **kwargs)
        metrics_manager = MetricsManager(metrics_dict)
        return metrics_manager

    def train_one_batch(self, input_batch: Dict[str, torch.Tensor]):
        """
        Train one batch of data and return metrics outputs

        Args:
            input_batch: A batch of inputs to train

        Returns:
            The metrics results
        """
        input_batch = {k: v.to(self.device) for k, v in input_batch.items() if isinstance(v, torch.Tensor)}
        outputs = self.model(input_batch)
        if "loss" not in outputs:
            raise ValueError("Model outputs must contain `loss`!")
        loss: torch.Tensor = outputs["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        results = self.metrics_manager.compute(outputs["logits"].detach().cpu(), input_batch["labels"].detach().cpu())
        results["loss"] = loss.item()

        return results

    def evaluate_one_batch(self, input_batch: Dict[str, torch.Tensor]):
        """
        Evaluate one batch of data and return metrics outputs

        Args:
            input_batch: A batch of inputs to train

        Returns:
            The metrics results
        """
        input_batch = {k: v.to(self.device) for k, v in input_batch.items() if isinstance(v, torch.Tensor)}
        outputs = self.model(input_batch)
        if "loss" not in outputs:
            raise ValueError("Model outputs must contain `loss`!")
        loss: torch.Tensor = outputs["loss"]

        results = self.metrics_manager.compute(outputs["logits"].detach().cpu(), input_batch["labels"].detach().cpu())
        results["loss"] = loss.item()

        return results

    def _one_training_loop(self, epoch_num: int):
        """
        Train the model for one epoch on the whole train dataset and verbose live metric values in the progress bar

        Args:
            epoch_num: number of the current epoch

        Returns:
            Metrics averages through the full iteration
        """
        self.metrics_manager.reset()
        self.model.train()
        with tqdm(
            self.train_dataloader,
            unit="batch",
            desc=f"Epoch: {epoch_num}/{self.num_train_epochs} ",
            bar_format=TQDM_BAR_FORMAT,
            ascii=" #",
        ) as iterator:
            for input_batch in iterator:
                results = self.train_one_batch(input_batch)
                self.metrics_manager.update(results)
                iterator.set_postfix(**self.metrics_manager.avg())
        return self.metrics_manager.avg()

    def evaluate(self):
        """
        Evaluates the model on the whole eval dataset and verbose live metric values in the progress bar

        Returns:
            Metrics averages through the full iteration
        """
        self.metrics_manager.reset()
        self.model.eval()
        with tqdm(
            self.eval_dataloader,
            unit="batch",
            desc="Evaluating... ",
            bar_format=TQDM_BAR_FORMAT,
            ascii=" #",
        ) as iterator:
            with torch.inference_mode():
                for input_batch in iterator:
                    results = self.evaluate_one_batch(input_batch)
                    self.metrics_manager.update(results)
                    iterator.set_postfix(**self.metrics_manager.avg())
        return self.metrics_manager.avg()

    def train(self):
        """
        The full training process like training, evaluation, logging and saving model checkpoints.
        """
        for epoch in range(0, self.num_train_epochs + 1):
            print()
            train_results = self._one_training_loop(epoch)
            eval_results = self.evaluate()
            self.lr_scheduler.step(eval_results["loss"])

            # tensorboard
            write_to_tensorboard(self.tensorboard, train_results, "train", epoch)
            write_to_tensorboard(self.tensorboard, eval_results, "val", epoch)

            # save checkpoint
            ckpt_save_path = os.path.join(self.config.checkpoints_dir, str(epoch))
            self.save(ckpt_save_path)

    def save(self, path: str):
        """
        Save the trainer and relevant files to a path.
        Files to save are train config, model weights, model config, preprocessor files and preprocessor config.

        Args:
            path: A directory to save everything
        """
        self.config.save(path, filename=self.trainer_config_file, subfolder=self.trainer_subfolder)
        self.model.save(path, save_config=True)
        self.train_dataset.config.save(path, filename=self.dataset_config_file, subfolder=self.trainer_subfolder)
        if hasattr(self.train_dataset, "tokenizer"):
            self.train_dataset.tokenizer.save(path)

    def push_to_hub(self, hub_path: str, commit_message: str = None):
        """
        Push everything to the Hub

        Args:
            hub_path: Path to hub
            commit_message: Commit message for the push
        """
        hub_path = resolve_pretrained_path(hub_path)
        cache_path = get_local_cache_path(hub_path, repo_type="model")

        self.save(cache_path)

        if not commit_message:
            commit_message = "Hezar: Upload with Trainer"

        upload_folder(
            repo_id=hub_path,
            folder_path=cache_path,
            repo_type="model",
            commit_message=commit_message,
        )
