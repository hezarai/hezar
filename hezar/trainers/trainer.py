import os
import random
import tempfile
from typing import Any, Dict, Tuple

import numpy as np
import torch
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..builders import build_metric
from ..configs import LRSchedulerConfig, MetricConfig, OptimizerConfig, TrainerConfig
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
from .trainer_utils import MetricsTracker


logger = get_logger(__name__)


optimizers = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}
lr_schedulers = {
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cosine_lr": torch.optim.lr_scheduler.CosineAnnealingLR,
}


class Trainer:
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
        compute_metrics=None,
        **kwargs,
    ):

        self.config = config

        self.device, self.device_type = self._prepare_device_and_type()
        self.autocast_dtype = torch.bfloat16 if self.device_type == "cpu" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp and self.device_type == "cuda")

        self._set_seed(self.config.seed)

        self.model = self._prepare_model(model)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator or self.train_dataset.data_collator
        self.train_dataloader, self.eval_dataloader = self._prepare_dataloaders()

        self.optimizer, self.lr_scheduler = self._prepare_optimizers(optimizer, lr_scheduler)

        self.metrics = self.setup_metrics()
        self.metrics_tracker = MetricsTracker(list(self.metrics.keys()))

        self.tensorboard = SummaryWriter(log_dir=self.config.log_dir)

    def _prepare_device_and_type(self) -> Tuple[str, str]:
        device = self.config.device if "cuda" in self.config.device and torch.cuda.is_available() else "cpu"
        device_type = "cuda" if "cuda" in device else "cpu"
        return device, device_type

    @staticmethod
    def _set_seed(seed) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _prepare_model(self, model: Model) -> Model:
        """
        Download the model from HuggingFace Hub if `init_weights_from` is given in the config. Load the model to the
        device and return it.
        :param model:
        :return:
        """
        if model is None:
            raise ValueError("`model` must be given to the Trainer!")
        hub_path = self.config.init_weights_from
        if hub_path is not None:
            local_path = hf_hub_download(hub_path, filename=model.model_filename, cache_dir=HEZAR_CACHE_DIR)
            model.load_state_dict(torch.load(local_path, map_location="cpu"))
        model.to(self.device)
        return model

    def _prepare_dataloaders(self):
        """
        Set up data loaders (train/eval) and return them.

        Returns:
             A tuple of train and eval dataloaders
        """
        if self.train_dataset is not None:
            train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.batch_size,
                collate_fn=self.data_collator,
                num_workers=self.config.num_dataloader_workers,
                drop_last=True,
                shuffle=True,
            )
        else:
            raise ValueError("Cannot create train dataloader because `train_dataset` is not given!")
        if self.eval_dataset is not None:
            eval_dataloader = DataLoader(
                dataset=self.eval_dataset,
                batch_size=self.config.batch_size,
                collate_fn=self.data_collator,
                num_workers=self.config.num_dataloader_workers,
                drop_last=True,
                shuffle=True,
            )
        else:
            logger.warning("Cannot create eval dataloader because `eval_dataset` is not given to the Trainer!")
            eval_dataloader = None

        return train_dataloader, eval_dataloader

    def _prepare_optimizers(self, optimizer: torch.optim.Optimizer = None, lr_scheduler=None):
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
            optimizer = optimizers[optimizer_name](self.model.parameters(), **optimizer_config)

            if lr_scheduler is None and scheduler_config is not None:
                if isinstance(scheduler_config, LRSchedulerConfig):
                    scheduler_config = scheduler_config.dict()
                scheduler_name = scheduler_config.pop("name")
                scheduler_config.pop("config_type", None)
                lr_scheduler = lr_schedulers[scheduler_name](optimizer, **scheduler_config)
        return optimizer, lr_scheduler

    def setup_metrics(self):
        metrics_dict = {}
        for metric in self.config.metrics:
            if isinstance(metric, str):
                if metric not in self.AVAILABLE_METRICS:
                    raise ValueError(f"Invalid metric `{metric}`! Available metrics: {self.AVAILABLE_METRICS}")
                metrics_dict[metric] = build_metric(metric)
            elif isinstance(metric, MetricConfig):
                metrics_dict[metric] = build_metric(metric.name, config=metric)
            else:
                raise ValueError(f"Invalid metric type `{type(metric)}`! Available metrics: {self.AVAILABLE_METRICS}")
        return metrics_dict

    def prepare_input_batch(self, input_batch):
        """
        Every operation required to prepare the inputs for model forward like moving to device, permutations, etc.
        Args:
            input_batch: Raw input batch from the dataloader

        Returns:
            The proper input batch required by model forward
        """
        # cast to device
        input_batch = {k: v.to(self.device) for k, v in input_batch.items() if isinstance(v, torch.Tensor)}
        return input_batch

    def amp_context_manager(self):
        """
        A smart context manager for mixed precision.

        Returns:
            A torch autocast context manager
        """
        return torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype, enabled=self.config.use_amp)

    def forward(self, input_batch):
        """
        Perform model forward on the input batch

        In special cases, one can override this method in their desired trainer.

        Args:
            input_batch: Input batch

        Returns:
            Model outputs
        """
        outputs = self.model(input_batch)
        return outputs

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform model forward and compute loss.

        This method must be implemented in other trainers.

        Args:
            logits: Logits from model outputs
            labels: Ground truth labels

        Returns:
            The loss tensor
        """
        raise NotImplementedError

    def compute_metrics(self, predictions, labels, **kwargs):
        """
        Compute metric values on the predictions and labels

        Args:
            predictions: A list of all predictions
            labels: A list of all labels

        Returns:
            A dictionary of the results for every metric specified by the trainer
        """
        return {}

    def training_step(self, input_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Train one batch of data and return loss and model outputs

        Args:
            input_batch: A batch of inputs to train

        Returns:
            Train step outputs including loss, logits, etc.
        """
        with self.amp_context_manager():
            outputs = self.forward(input_batch)
            loss = self.compute_loss(outputs["logits"], input_batch["labels"])

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        outputs["loss"] = loss.item() if isinstance(loss, torch.Tensor) else loss

        return outputs

    def evaluation_step(self, input_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Evaluate one batch of data and return loss and model outputs

        Args:
            input_batch: A batch of inputs to evaluate

        Returns:
            Evaluation step outputs including loss, logits, etc.
        """
        with self.amp_context_manager():
            outputs = self.forward(input_batch)
            loss = self.compute_loss(outputs["logits"], input_batch["labels"])

        outputs["loss"] = loss.item() if isinstance(loss, torch.Tensor) else loss

        return outputs

    def inner_training_loop(self, epoch_num: int):
        """
        Train the model for one epoch on the whole train dataset and verbose live metric values in the progress bar

        Args:
            epoch_num: Number of the current epoch

        Returns:
            Metrics averages through the full iteration
        """
        self.metrics_tracker.reset()
        self.model.train()
        with tqdm(
            self.train_dataloader,
            unit="batch",
            desc=f"Epoch: {epoch_num}/{self.config.num_epochs} ",
            bar_format=TQDM_BAR_FORMAT,
            ascii=" #",
        ) as iterator:
            for step, input_batch in enumerate(iterator):
                input_batch = self.prepare_input_batch(input_batch)
                # Training on one batch
                outputs = self.training_step(input_batch)
                # Compute metrics
                training_results = self.compute_metrics(
                    outputs["logits"].detach().cpu().numpy(),
                    input_batch["labels"].detach().cpu().numpy(),
                )
                training_results["loss"] = outputs["loss"]
                # Gather outputs for metrics
                self.metrics_tracker.update(training_results)
                iterator.set_postfix(**self.metrics_tracker.avg())

        return self.metrics_tracker.avg()

    def evaluate(self):
        """
        Evaluates the model on the whole eval dataset and verbose live metric values in the progress bar

        Returns:
            Evaluation results
        """
        self.metrics_tracker.reset()
        self.model.eval()
        with tqdm(
            self.eval_dataloader,
            unit="batch",
            desc="Evaluating... ",
            bar_format=TQDM_BAR_FORMAT,
            ascii=" #",
        ) as iterator:
            with torch.inference_mode():
                for step, input_batch in enumerate(iterator):
                    input_batch = self.prepare_input_batch(input_batch)
                    # Evaluation on one batch
                    outputs = self.evaluation_step(input_batch)
                    # Compute metrics
                    evaluation_results = self.compute_metrics(
                        outputs["logits"].detach().cpu().numpy(),
                        input_batch["labels"].detach().cpu().numpy(),
                )
                    evaluation_results["loss"] = outputs["loss"]
                    # Gather outputs for metrics
                    self.metrics_tracker.update(evaluation_results)
                    iterator.set_postfix(**self.metrics_tracker.avg())

        return self.metrics_tracker.avg()

    def train(self):
        """
        The full training process like training, evaluation, logging and saving model checkpoints.
        """
        for epoch in range(1, self.config.num_epochs + 1):
            print()
            self.inner_training_loop(epoch)
            evaluation_results = self.evaluate()
            self.lr_scheduler.step(evaluation_results["loss"])

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
