import inspect
import os
import random
from typing import Any, Callable, Dict, Mapping, Tuple, Union

import numpy as np
import torch
from huggingface_hub import create_repo, hf_hub_download, upload_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..configs import TrainerConfig
from ..constants import (
    DEFAULT_DATASET_CONFIG_FILE,
    DEFAULT_TRAINER_CONFIG_FILE,
    DEFAULT_TRAINER_CSV_LOG_FILE,
    DEFAULT_TRAINER_SUBFOLDER,
    HEZAR_CACHE_DIR,
    TQDM_BAR_FORMAT,
    TaskType,
)
from ..data.datasets import Dataset
from ..models import Model
from ..preprocessors import Preprocessor, PreprocessorsContainer
from ..utils import Logger, sanitize_params_for_fn
from .metrics_handlers import (
    AudioClassificationMetricsHandler,
    Image2TextMetricHandler,
    MetricsHandler,
    SequenceLabelingMetricsHandler,
    SpeechRecognitionMetricsHandler,
    TextClassificationMetricsHandler,
    TextGenerationMetricsHandler,
)
from .trainer_utils import CSVLogger, write_to_tensorboard


logger = Logger(__name__)

optimizers = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}
lr_schedulers = {
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cosine_lr": torch.optim.lr_scheduler.CosineAnnealingLR,
}

task_to_metrics_handlers_mapping = {
    TaskType.TEXT_CLASSIFICATION: TextClassificationMetricsHandler,
    TaskType.SEQUENCE_LABELING: SequenceLabelingMetricsHandler,
    TaskType.IMAGE2TEXT: Image2TextMetricHandler,
    TaskType.SPEECH_RECOGNITION: SpeechRecognitionMetricsHandler,
    TaskType.AUDIO_CLASSIFICATION: AudioClassificationMetricsHandler,
    TaskType.TEXT_GENERATION: TextGenerationMetricsHandler,

}


class Trainer:
    """
    Base trainer class for training all Hezar models and all tasks. Usually you can use this class as-is, but for special
    cases you can also override any of the core methods in your own custom Trainer.

    Args:
        model ([`Model`] or `torch.nn.Module`): The main model to train and evaluate
        config (TrainerConfig): Training configuration and parameters
        train_dataset (Dataset): Train dataset
        eval_dataset (Dataset): Evaluation dataset
        data_collator: Collate function, usually included in the dataset object itself
        preprocessor: Preprocessor object(s)
        metrics_handler: Optional metrics handler
        optimizer (optim.Optimizer): Model optimizer
        lr_scheduler: Optional learning-rate scheduler

    """

    trainer_subfolder = DEFAULT_TRAINER_SUBFOLDER
    trainer_config_file = DEFAULT_TRAINER_CONFIG_FILE
    trainer_csv_log_file = DEFAULT_TRAINER_CSV_LOG_FILE
    dataset_config_file = DEFAULT_DATASET_CONFIG_FILE
    AVAILABLE_METRICS = []
    default_optimizer = "adam"
    default_lr_scheduler = None

    def __init__(
        self,
        model: Model = None,
        config: TrainerConfig = None,
        train_dataset: Dataset = None,
        eval_dataset: Dataset = None,
        data_collator: Callable = None,
        preprocessor: Union[Preprocessor, PreprocessorsContainer] = None,
        metrics_handler: MetricsHandler = None,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler=None,
    ):
        self.config = config

        self.device, self.device_type = self._prepare_device_and_type()
        self.autocast_dtype = torch.bfloat16 if self.device_type == "cpu" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp and self.device_type == "cuda")

        self._set_seed(self.config.seed)

        self.model = self._prepare_model(model)
        self.model.preprocessor = preprocessor

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator or self.train_dataset.data_collator
        self.train_dataloader, self.eval_dataloader = self._prepare_dataloaders()

        self.optimizer, self.lr_scheduler = self._prepare_optimizers(optimizer, lr_scheduler)

        self.metrics_handler = metrics_handler or self._prepare_metrics_handler()

        self.tensorboard = SummaryWriter(log_dir=self.config.logs_dir)
        self.csv_logger = CSVLogger(logs_dir=self.config.logs_dir, csv_filename=self.trainer_csv_log_file)

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
        """
        if model is None:
            raise ValueError("`model` must be given to the Trainer!")
        hub_path = self.config.init_weights_from
        if hub_path is not None:
            if os.path.isdir(hub_path):
                model_path = os.path.join(hub_path, model.model_filename)
            else:
                model_path = hf_hub_download(
                    hub_path,
                    filename=model.model_filename,
                    cache_dir=HEZAR_CACHE_DIR,
                    resume_download=True,
                )
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.to(self.device)
        return model

    def _prepare_dataloaders(self) -> Tuple[DataLoader, Union[DataLoader, None]]:
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
                batch_size=self.config.eval_batch_size or self.config.batch_size,
                collate_fn=self.data_collator,
                num_workers=self.config.num_dataloader_workers,
                drop_last=True,
                shuffle=True,
            )
        else:
            logger.warning(
                "Cannot create eval dataloader because `eval_dataset` is not given to the Trainer! "
                "Setting eval_dataloader to None..."
            )
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
            optimizer_type = self.config.optimizer or self.default_optimizer
            optimizer = optimizers[optimizer_type](
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            if lr_scheduler is None:
                scheduler_name = self.config.scheduler or self.default_lr_scheduler
                if scheduler_name is None:
                    lr_scheduler = None
                else:
                    lr_scheduler = lr_schedulers[scheduler_name](optimizer, verbose=True)
        return optimizer, lr_scheduler

    def _prepare_metrics_handler(self):
        metrics_handler_cls = task_to_metrics_handlers_mapping[self.config.task]
        metrics_handler = metrics_handler_cls(
            metrics=self.config.metrics,
            trainer=self,
        )
        return metrics_handler

    def prepare_input_batch(self, input_batch) -> Dict[str, torch.Tensor]:
        """
        Every operation required to prepare the inputs for model forward like moving to device, permutations, etc.
        Args:
            input_batch: Raw input batch from the dataloader

        Returns:
            The proper input batch required by model forward
        """
        # cast to device
        input_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_batch.items()}
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
        if isinstance(input_batch, torch.Tensor):
            outputs = self.model(input_batch)
        elif isinstance(input_batch, Dict):
            forward_inputs = sanitize_params_for_fn(self.model.forward, input_batch)
            outputs = self.model(**forward_inputs)
        else:
            raise ValueError(
                f"`input_batch` must be a tensor or a dict-like object containing key/value pairs of tensors, "
                f"but got {type(input_batch)}"
            )
        if not isinstance(outputs, Dict):
            raise ValueError(f"Model outputs must be dict-like not `{type(outputs)}`")

        return outputs

    def compute_loss(self, model_outputs: Dict, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute loss from model outputs

        This method must be implemented in other trainers.

        Args:
            model_outputs: Logits from model outputs
            labels: Ground truth labels

        Returns:
            The loss tensor
        """
        compute_loss_inputs = sanitize_params_for_fn(self.model.compute_loss, model_outputs, **kwargs)
        compute_loss_inputs["labels"] = labels

        loss = self.model.compute_loss(**compute_loss_inputs)

        return loss

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
            loss = self.compute_loss(outputs, **input_batch)

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
            loss = self.compute_loss(outputs, **input_batch)
            if self.model.is_generative and self.config.evaluate_with_generate:
                generate_inputs = sanitize_params_for_fn(self.model.generate, input_batch)
                generated_ids = self.model.generate(**generate_inputs)
                outputs["logits"] = generated_ids

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
        losses_sum = 0.0
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
                losses_sum += outputs["loss"]
                # Gather outputs for metrics
                avg_loss = losses_sum / (step + 1)
                iterator.set_postfix(loss=avg_loss)

        return {"loss": avg_loss}

    def evaluate(self):
        """
        Evaluates the model on the whole eval dataset and verbose live metric values in the progress bar

        Returns:
            Evaluation results
        """
        self.metrics_handler.tracker.reset()
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
                    logits = outputs["logits"].detach().cpu().numpy()
                    labels = input_batch["labels"].detach().cpu().numpy()
                    # Compute metrics
                    evaluation_results = self.metrics_handler.compute_metrics(logits, labels)
                    evaluation_results["loss"] = outputs["loss"]
                    # Gather outputs for metrics
                    self.metrics_handler.tracker.update(evaluation_results)
                    iterator.set_postfix(**self.metrics_handler.tracker.avg())

        return self.metrics_handler.tracker.avg()

    def train(self):
        """
        The full training process like training, evaluation, logging and saving model checkpoints.
        """
        for epoch in range(1, self.config.num_epochs + 1):
            print()
            training_results = self.inner_training_loop(epoch)
            evaluation_results = self.evaluate()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(evaluation_results["loss"])

            current_step_logs = {
                "step": epoch,
                "training_results": training_results,
                "evaluation_results": evaluation_results,
            }

            self.log(current_step_logs)

            # maybe save checkpoint
            if epoch % self.config.save_freq == 0:
                ckpt_save_path = os.path.join(self.config.checkpoints_dir, str(epoch))
                self.save(ckpt_save_path)

    def log(self, logs: Dict[str, Any]):
        """
        Log metrics results
        """
        step = logs["step"]
        training_results = logs["training_results"]
        evaluation_results = logs["evaluation_results"]

        train_logs = {f"train.{metric_name}": value for metric_name, value in training_results.items()}
        evaluation_logs = {f"evaluation.{metric_name}": value for metric_name, value in evaluation_results.items()}

        # Log to tensorboard
        write_to_tensorboard(self.tensorboard, train_logs, step)
        write_to_tensorboard(self.tensorboard, evaluation_logs, step)

        # Log to CSV
        self.csv_logger.write({**train_logs, **evaluation_logs}, step)

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
        self.model.save(path, filename=model_filename, config_filename=model_config_filename)
        self.train_dataset.config.save(path, filename=dataset_config_file, subfolder=subfolder)

    def push_to_hub(
        self,
        repo_id: str,
        config_filename: str = None,
        push_model: bool = True,
        push_logs: bool = True,
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
            push_model: Whether to push the model or not
            push_logs: Whether to push training logs or not
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

        if not commit_message:
            commit_message = "Hezar: Upload training files"

        # upload train files
        self.config.push_to_hub(
            repo_id,
            filename=config_filename,
            subfolder=subfolder,
            private=private,
            commit_message=commit_message,
        )
        self.train_dataset.config.push_to_hub(
            repo_id,
            filename=dataset_config_file,
            subfolder=subfolder,
            private=private,
            commit_message=commit_message
        )

        # upload model
        if push_model:
            self.model.push_to_hub(
                repo_id,
                filename=model_filename,
                config_filename=model_config_filename,
                commit_message=commit_message,
                private=private,
            )

        if push_logs:
            upload_file(
                path_or_fileobj=self.csv_logger.save_path,
                path_in_repo=os.path.join(self.trainer_subfolder, self.trainer_csv_log_file),
                repo_id=repo_id,
                commit_message=commit_message,
            )
