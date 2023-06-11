import os
import random
import tempfile
from typing import Dict

import numpy as np
import torch
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, F1Score, Precision
from tqdm import tqdm

from ..builders import build_optimizer, build_scheduler
from ..configs import LRSchedulerConfig, OptimizerConfig, TrainConfig
from ..constants import (
    DEFAULT_DATASET_CONFIG_FILE,
    DEFAULT_TRAINER_CONFIG_FILE,
    DEFAULT_TRAINER_SUBFOLDER,
    HEZAR_CACHE_DIR,
    TQDM_BAR_FORMAT,
)
from ..data.datasets import Dataset
from ..models import Model
from ..utils import get_logger
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
        model ([`Model`] or `torch.nn.Module`): The main model to train and evaluate
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

        self.device, self.device_type = self._set_device_and_type()
        self.autocast_dtype = torch.bfloat16 if self.device_type == "cpu" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp and self.device_type == "cuda")

        self.set_seed(self.config.seed)

        self.model = self._init_model_weights(model).to(self.device)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.num_labels = self.train_dataset.num_labels  # noqa

        self.train_dataloader, self.eval_dataloader = self._setup_dataloaders()

        self.optimizer, self.lr_scheduler = self._setup_optimizers(optimizer, lr_scheduler)

        self.metrics_manager = self._setup_metrics_manager(self.config.metrics)

        self.tensorboard = SummaryWriter(log_dir=self.config.log_dir)

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _init_model_weights(self, model):
        hub_path = self.config.init_weights_from
        local_path = hf_hub_download(hub_path, filename=model.model_filename, cache_dir=HEZAR_CACHE_DIR)
        model.load_state_dict(torch.load(local_path, map_location="cpu"))
        return model

    def _set_device_and_type(self):
        device = self.config.device if "cuda" in self.config.device and torch.cuda.is_available() else "cpu"
        device_type = "cuda" if "cuda" in device else "cpu"
        return device, device_type

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
            optimizer_config = self.config.optimizer

            # convert to dict so that we can pop some values
            if isinstance(optimizer_config, OptimizerConfig):
                optimizer_config = optimizer_config.dict()

            optimizer_name = optimizer_config.pop("name")
            scheduler_config = optimizer_config.pop("scheduler")

            optimizer_config.pop("config_type", None)
            optimizer = build_optimizer(
                optimizer_name,
                self.model.parameters(),
                **optimizer_config,
            )

            if lr_scheduler is None and scheduler_config is not None:
                if isinstance(scheduler_config, LRSchedulerConfig):
                    scheduler_config = scheduler_config.dict()
                scheduler_name = scheduler_config.pop("name")
                scheduler_config.pop("config_type", None)
                lr_scheduler = build_scheduler(
                    scheduler_name,
                    optimizer,
                    **scheduler_config,
                )
        return optimizer, lr_scheduler

    def _setup_metrics_manager(self, metrics: Dict[str, Dict]) -> MetricsManager:
        """
        Set up metrics manager to track and update metrics like loss, accuracy, f1, etc.

        Args:
            metrics: A dict of metrics names and their kwargs {metric_name: **kwargs}

        Returns:
             A MetricsManager instance
        """
        metrics_dict = {"loss": None}
        for name, kwargs in metrics.items():
            metrics_dict[name] = METRICS_MAP[name](num_classes=self.num_labels, **kwargs)
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

    def inner_training_loop(self, epoch_num: int):
        """
        Train the model for one epoch on the whole train dataset and verbose live metric values in the progress bar

        Args:
            epoch_num: Number of the current epoch

        Returns:
            Metrics averages through the full iteration
        """
        self.metrics_manager.reset()
        self.model.train()
        with tqdm(
            self.train_dataloader,
            unit="batch",
            desc=f"Epoch: {epoch_num}/{self.config.num_epochs} ",
            bar_format=TQDM_BAR_FORMAT,
            ascii=" #",
        ) as iterator:
            for input_batch in iterator:
                results = self.training_step(input_batch)
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
                    results = self.evaluation_step(input_batch)
                    self.metrics_manager.update(results)
                    iterator.set_postfix(**self.metrics_manager.avg())
        return self.metrics_manager.avg()

    def train(self):
        """
        The full training process like training, evaluation, logging and saving model checkpoints.
        """
        for epoch in range(1, self.config.num_epochs + 1):
            print()
            train_results = self.inner_training_loop(epoch)
            eval_results = self.evaluate()
            self.lr_scheduler.step(eval_results["loss"])

            # tensorboard
            write_to_tensorboard(self.tensorboard, train_results, "train", epoch)
            write_to_tensorboard(self.tensorboard, eval_results, "val", epoch)

            # maybe save checkpoint
            if epoch % self.config.save_freq == 0:
                ckpt_save_path = os.path.join(self.config.checkpoints_dir, str(epoch))
                self.save(ckpt_save_path)

    def save(
        self,
        path: str,
        config_filename=None,
        model_filename=None,
        model_config_filename=None,
        subfolder=None,
        dataset_config_file=None,
    ):
        """
        Save the trainer and relevant files to a path.

        Files to save are train config, model weights, model config, preprocessor files and preprocessor config.

        Args:
            path: A directory to save everything
            config_filename: Config filename
            model_filename: Model file name
            model_config_filename: Model config file name
            subfolder: Optional sub-folder
            dataset_config_file: Dataset config filename
        """
        config_filename = config_filename or self.trainer_config_file
        subfolder = subfolder or self.trainer_subfolder
        dataset_config_file = dataset_config_file or self.dataset_config_file

        self.config.save(path, filename=config_filename, subfolder=subfolder)
        self.model.save(path, filename=model_filename, config_filename=model_config_filename, save_config=True)
        self.train_dataset.config.save(path, filename=dataset_config_file, subfolder=subfolder)
        if hasattr(self.train_dataset, "tokenizer"):
            self.train_dataset.tokenizer.save(path)

    def push_to_hub(
        self,
        repo_id: str,
        config_filename: str = None,
        model_filename: str = None,
        model_config_filename: str = None,
        subfolder: str = None,
        dataset_config_filename: str = None,
        commit_message: str = None,
        private: bool = False,
    ):
        """
        Push everything to the Hub

        Args:
            repo_id: Path to hub
            config_filename: Trainer config file name
            model_filename: Model file name
            model_config_filename: Model config file name
            subfolder: Path to Trainer files
            dataset_config_filename: Dataset config file name
            commit_message: Commit message for the push
            private: Whether to create a private repo if it doesn't exist already
        """
        config_filename = config_filename or self.trainer_config_file
        subfolder = subfolder or self.trainer_subfolder
        dataset_config_file = dataset_config_filename or self.dataset_config_file

        # create remote repo
        create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
        # save to tmp and prepare for push
        cache_path = tempfile.mkdtemp()

        self.save(
            cache_path,
            config_filename=config_filename,
            model_filename=model_filename,
            model_config_filename=model_config_filename,
            subfolder=subfolder,
            dataset_config_file=dataset_config_file,
        )

        if not commit_message:
            commit_message = "Hezar: Upload training files"

        upload_folder(
            repo_id=repo_id,
            folder_path=cache_path,
            repo_type="model",
            commit_message=commit_message,
        )
