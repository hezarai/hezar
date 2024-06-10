from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Callable, Dict

import pandas as pd
import torch
from huggingface_hub import create_repo, hf_hub_download, upload_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ..configs import TrainerConfig
from ..constants import (
    DEFAULT_DATASET_CONFIG_FILE,
    DEFAULT_LR_SCHEDULER_FILE,
    DEFAULT_OPTIMIZER_FILE,
    DEFAULT_TRAINER_CONFIG_FILE,
    DEFAULT_TRAINER_CSV_LOG_FILE,
    DEFAULT_TRAINER_STATE_FILE,
    DEFAULT_TRAINER_SUBFOLDER,
    HEZAR_CACHE_DIR,
    TQDM_BAR_FORMAT,
    Backends,
    LRSchedulerType,
    OptimizerType,
    TaskType,
)
from ..data import Dataset, RangedSampler
from ..models import Model
from ..preprocessors import Preprocessor, PreprocessorsContainer
from ..utils import (
    Logger,
    colorize_text,
    dataloader_worker_init_fn,
    is_backend_available,
    sanitize_function_parameters,
    set_seed,
    verify_dependencies,
)
from .metrics_handlers import (
    Image2TextMetricHandler,
    MetricsHandler,
    SequenceLabelingMetricsHandler,
    SpeechRecognitionMetricsHandler,
    TextClassificationMetricsHandler,
    TextGenerationMetricsHandler,
)
from .trainer_utils import (
    AverageMeter,
    CSVLogger,
    TrainerState,
    get_distributed_logger,
    resolve_logdir,
    write_to_tensorboard,
)


if is_backend_available(Backends.ACCELERATE):
    from accelerate import Accelerator
else:
    raise ImportError("The package `accelerate` needs to be installed to use `Trainer`!")

logger = Logger(__name__)

optimizers = {
    OptimizerType.ADAM: torch.optim.Adam,
    OptimizerType.ADAMW: torch.optim.AdamW,
    OptimizerType.SDG: torch.optim.SGD,
}
lr_schedulers = {
    LRSchedulerType.LAMBDA: torch.optim.lr_scheduler.LambdaLR,
    LRSchedulerType.STEP: torch.optim.lr_scheduler.StepLR,
    LRSchedulerType.MULTI_STEP: torch.optim.lr_scheduler.MultiStepLR,
    LRSchedulerType.ONE_CYCLE: torch.optim.lr_scheduler.OneCycleLR,
    LRSchedulerType.LINEAR: torch.optim.lr_scheduler.LinearLR,
    LRSchedulerType.EXPONENTIAL: torch.optim.lr_scheduler.ExponentialLR,
    LRSchedulerType.CYCLIC: torch.optim.lr_scheduler.CyclicLR,
    LRSchedulerType.SEQUENTIAL: torch.optim.lr_scheduler.SequentialLR,
    LRSchedulerType.POLYNOMIAL: torch.optim.lr_scheduler.PolynomialLR,
    LRSchedulerType.COSINE_ANEALING: torch.optim.lr_scheduler.CosineAnnealingLR,
}

task_to_metrics_handlers_mapping = {
    TaskType.TEXT_CLASSIFICATION: TextClassificationMetricsHandler,
    TaskType.SEQUENCE_LABELING: SequenceLabelingMetricsHandler,
    TaskType.IMAGE2TEXT: Image2TextMetricHandler,
    TaskType.SPEECH_RECOGNITION: SpeechRecognitionMetricsHandler,
    TaskType.TEXT_GENERATION: TextGenerationMetricsHandler,
}


class Trainer:
    """
    Base trainer class for training all Hezar models and all tasks. Usually you can use this class as-is, but for special
    cases you can also override any of the core methods in your own custom Trainer.

    Args:
        model (`Model` | `torch.nn.Module`): The main model to train and evaluate
        config (TrainerConfig): Training configuration and parameters
        train_dataset (Dataset): Train dataset
        eval_dataset (Dataset): Evaluation dataset
        data_collator: Collate function, usually included in the dataset object itself
        preprocessor: Preprocessor object(s)
        metrics_handler: Optional metrics handler
        optimizer (optim.Optimizer): Model optimizer
        lr_scheduler: Optional learning-rate scheduler
        accelerator (Accelerator) : Accelerator object for a customized distributed environment
    """

    trainer_subfolder = DEFAULT_TRAINER_SUBFOLDER
    trainer_config_file = DEFAULT_TRAINER_CONFIG_FILE
    trainer_csv_log_file = DEFAULT_TRAINER_CSV_LOG_FILE
    dataset_config_file = DEFAULT_DATASET_CONFIG_FILE
    trainer_state_file = DEFAULT_TRAINER_STATE_FILE
    optimizer_file = DEFAULT_OPTIMIZER_FILE
    lr_scheduler_file = DEFAULT_LR_SCHEDULER_FILE
    default_optimizer = OptimizerType.ADAM
    default_lr_scheduler = None
    _required_backends = [Backends.ACCELERATE]

    def __init__(
        self,
        model: Model,
        config: TrainerConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        data_collator: Callable = None,
        preprocessor: Preprocessor | PreprocessorsContainer = None,
        metrics_handler: MetricsHandler = None,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler=None,
        accelerator: "Accelerator" = None,
    ):
        # Check if all required dependencies are installed
        verify_dependencies(self._required_backends)

        # Configuration
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and not self.config.use_cpu else "cpu"

        # Setup hardware acceleration controller
        self.accelerator = accelerator or Accelerator(
            mixed_precision=self.config.mixed_precision,
            cpu=True if self.device == "cpu" else False,
            step_scheduler_with_optimizer=False,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        # Setup logger
        self.logger = get_distributed_logger(__name__)

        # Configure datasets and data loaders
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator or getattr(self.train_dataset, "data_collator", None)

        self.num_batches = math.ceil(len(self.train_dataset) / self.config.batch_size)
        self.total_steps = min(
            self.config.max_steps or self.num_batches * self.config.num_epochs,
            self.num_batches * self.config.num_epochs
        )
        self.steps_in_epoch = min(self.num_batches, self.total_steps)
        self.config.save_steps = self.steps_in_epoch if not self.config.save_steps else self.config.save_steps
        self.saves_in_epoch = math.floor(self.steps_in_epoch / self.config.save_steps)

        # Setup checkpoint and state handler
        self.checkpoints_dir = os.path.join(self.config.output_dir, self.config.checkpoints_dir)
        self.state = self._create_trainer_state(self.config.resume_from_checkpoint)
        self.logs_dir = self.state.logs_dir

        # Set determinism
        set_seed(self.config.seed)

        # Setup model and preprocessor(s)
        self.model = self._setup_model(model)
        if model.preprocessor is None:
            if preprocessor is not None:
                model.preprocessor = preprocessor
            else:
                raise ValueError(
                    "You must set a preprocessor for the model or pass the preprocessor parameter to the Trainer!"
                )

        # Setup optimizer and (optionally) lr scheduler
        self.optimizer, self.lr_scheduler = self._create_optimizers(optimizer, lr_scheduler)

        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

        # Setup metrics handler and inner trackers for the trainer
        self.metrics_handler = metrics_handler or self._create_metrics_handler()
        self.train_loss_tracker = AverageMeter(
            name="train.loss",
            avg=self.state.loss_tracker_avg,
            sum=self.state.loss_tracker_sum,
            count=self.state.global_step,
        )

        # Setup logging properties
        self.tensorboard = SummaryWriter(log_dir=self.logs_dir)
        self.csv_logger = CSVLogger(logs_dir=self.logs_dir, csv_filename=self.trainer_csv_log_file)

    def _create_trainer_state(self, checkpoint: str = None):
        """
        Create Trainer's state attribute.

        Args:
            checkpoint: Optional checkpoint path or bool value to load the state from there

        Returns:
            A TrainerState object
        """
        trainer_state_path = os.path.join(self.checkpoints_dir, self.trainer_state_file)
        if checkpoint is not None and os.path.isfile(trainer_state_path):
            state = TrainerState.load(trainer_state_path)
            # Overwrite some fields if checkpoint is a path
            if os.path.isdir(checkpoint):
                step = os.path.basename(checkpoint)
                state.global_step = int(step) if step.isdigit() else self.state.global_step
                state.epoch = math.ceil(state.global_step / self.steps_in_epoch)
                state.epoch_step = state.global_step % self.steps_in_epoch

            if state.epoch_step == 0:
                state.epoch += 1
        else:
            state = TrainerState(
                epoch=1,
                total_epochs=self.config.num_epochs,
                metric_for_best_checkpoint=self.config.metric_for_best_model,
                logs_dir=resolve_logdir(os.path.join(self.config.output_dir, self.config.logs_dir)),
            )

        return state

    def _resolve_checkpoint_path(self, checkpoint: str | bool):
        if isinstance(checkpoint, bool) and checkpoint:
            checkpoint_name = str(self.state.global_step).zfill(len(str(self.total_steps)))
            checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)
        elif os.path.isdir(checkpoint):
            checkpoint_path = checkpoint
        else:
            raise ValueError(f"Checkpoint `{checkpoint}` is either invalid or does not exist!")

        return checkpoint_path

    def _setup_model(self, model: Model) -> Model:
        """
        Create and load the weights for the model. The weights will be loaded to the model depending
        on `config.resume_from_checkpoint` or `config.init_weights_from`.
        """
        if model is None:
            raise ValueError("`model` must be given to the Trainer!")

        # Maybe load from checkpoint
        if self.config.resume_from_checkpoint:
            checkpoint_path = self._resolve_checkpoint_path(self.config.resume_from_checkpoint)
            model_path = os.path.join(checkpoint_path, model.model_filename)
            if os.path.isdir(checkpoint_path) and os.path.isfile(model_path):
                model.load_state_dict(torch.load(model_path))
                self.logger.info(f"Resuming training from checkpoint at `{checkpoint_path}`")
        # Maybe load from pretrained weights locally or from a hub repo
        elif self.config.init_weights_from is not None and self.state.global_step == 0 and self.state.epoch == 1:
            if self.config.init_weights_from is not None:
                if os.path.isdir(self.config.init_weights_from):
                    model_path = os.path.join(self.config.init_weights_from, model.model_filename)
                else:
                    model_path = hf_hub_download(
                        self.config.init_weights_from,
                        filename=model.model_filename,
                        cache_dir=HEZAR_CACHE_DIR,
                        resume_download=True,
                    )
                model.load_state_dict(torch.load(model_path))

        return model

    def create_train_dataloader(self, dataset) -> DataLoader:
        """
        Create train data loader using a ranged sampler that can handle slicing data, shuffling, etc.
        """
        start_index = self.state.epoch_step * self.config.batch_size
        sampler = RangedSampler(
            dataset,
            self.config.batch_size,
            start_index=start_index,
            shuffle=self.config.dataloader_shuffle,
            seed=self.config.seed,
            drop_last=False,
        )
        worker_init_fn = dataloader_worker_init_fn(self.config.seed)

        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.data_collator,
            sampler=sampler,
            num_workers=self.config.num_dataloader_workers,
            worker_init_fn=worker_init_fn,
        )

        train_dataloader = self.accelerator.prepare(train_dataloader)

        return train_dataloader

    def create_eval_dataloader(self, dataset) -> DataLoader:
        """
        Create eval data loader using a ranged sampler that can handle slicing data, shuffling, etc.
        """
        sampler = RangedSampler(
            dataset,
            self.config.eval_batch_size or self.config.batch_size,
            start_index=0,  # We don't support resumption for evaluation so we always start from zero
            shuffle=self.config.dataloader_shuffle,
            seed=self.config.seed,
            drop_last=False,
        )
        worker_init_fn = dataloader_worker_init_fn(self.config.seed)

        eval_dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.eval_batch_size or self.config.batch_size,
            collate_fn=self.data_collator,
            sampler=sampler,
            num_workers=self.config.num_dataloader_workers,
            worker_init_fn=worker_init_fn,
        )

        eval_dataloader = self.accelerator.prepare(eval_dataloader)

        return eval_dataloader

    def _create_optimizers(self, optimizer: torch.optim.Optimizer = None, lr_scheduler=None):
        """
        Set up the optimizer and lr lr_scheduler if they're not already given

        Args:
            optimizer: If None do nothing and return it, otherwise build it using the train config
            lr_scheduler: If None do nothing and return it, otherwise build it using the train config

        Returns:
            Optimizer and lr_scheduler
        """
        if optimizer is None:
            optimizer_type = self.config.optimizer or self.default_optimizer
            optimizer = optimizers[optimizer_type](
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            if lr_scheduler is None:
                scheduler_name = self.config.lr_scheduler or self.default_lr_scheduler
                scheduler_kwargs = self.config.lr_scheduler_kwargs or {}
                if scheduler_name is None:
                    lr_scheduler = None
                else:
                    lr_scheduler = lr_schedulers[scheduler_name](optimizer, **scheduler_kwargs, verbose=True)

        if self.config.resume_from_checkpoint:
            checkpoint_path = self._resolve_checkpoint_path(self.config.resume_from_checkpoint)
            if os.path.isdir(checkpoint_path):
                optimizer_path = os.path.join(checkpoint_path, self.trainer_subfolder, self.optimizer_file)
                lr_scheduler_path = os.path.join(checkpoint_path, self.trainer_subfolder, self.lr_scheduler_file)
                if os.path.isfile(optimizer_path):
                    optimizer.load_state_dict(torch.load(optimizer_path))
                if lr_scheduler is not None and os.path.isfile(lr_scheduler_path):
                    lr_scheduler.load_state_dict(torch.load(lr_scheduler_path))

        return optimizer, lr_scheduler

    def _create_metrics_handler(self):
        """
        Setup MetricsHandler instance for the trainer

        Returns:
            A MetricsHandler subclass instance based on self.config.task
        """
        metrics_handler_cls = task_to_metrics_handlers_mapping[self.config.task]
        metrics_handler = metrics_handler_cls(metrics=self.config.metrics, trainer=self)  # noqa
        return metrics_handler

    def load_csv_logs(self, logs_dir=None):
        """
        Load the CSV log file
        Args:
            logs_dir: Path to logs directory, defaults to self.config.logs_dir

        Returns:
            Logs dictionary
        """
        logs_dir = logs_dir or self.logs_dir
        csv_path = os.path.join(logs_dir, self.trainer_csv_log_file)
        logs = pd.read_csv(csv_path)
        return logs.to_dict()

    def prepare_input_batch(self, input_batch) -> Dict[str, torch.Tensor]:
        """
        Every operation required to prepare the inputs for model forward like moving to device, permutations, etc.

        Args:
            input_batch: Raw input batch from the dataloader

        Returns:
            The proper input batch required by model forward
        """
        # Put inputs on device manually if accelerator is not available, otherwise it's taken care of by the accelerator
        if self.accelerator is None:
            input_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_batch.items()}

        return input_batch

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
            forward_inputs = sanitize_function_parameters(self.model.forward, input_batch)
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

        Args:
            model_outputs: Logits from model outputs
            labels: Ground truth labels

        Returns:
            The loss tensor
        """
        compute_loss_inputs = sanitize_function_parameters(self.model.compute_loss, model_outputs, **kwargs)
        compute_loss_inputs["labels"] = labels

        loss = self.model.compute_loss(**compute_loss_inputs)

        return loss

    def optimization_step(self):
        """
        Perform optimization step
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def lr_scheduler_step(self, metrics=None):
        """
        Perform the learning rate scheduling step

        Args:
            metrics: one or multiple values that the scheduler watches to either perform step function or not. Only
             works for `ReduceLROnPlateau`.
        """
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, lr_schedulers[LRSchedulerType.REDUCE_ON_PLATEAU]):
                if metrics:
                    self.lr_scheduler.step(metrics)
            else:
                self.lr_scheduler.step()

    def training_step(self, input_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Train one batch of data and return loss and model outputs

        Args:
            input_batch: A batch of inputs to train

        Returns:
            Train step outputs including loss, logits, etc.
        """
        with self.accelerator.autocast():
            outputs = self.forward(input_batch)

        loss = self.compute_loss(outputs, **input_batch)

        self.accelerator.backward(loss)

        outputs["loss"] = loss.detach()

        return outputs

    def evaluation_step(self, input_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Evaluate one batch of data and return loss and model outputs

        Args:
            input_batch: A batch of inputs to evaluate

        Returns:
            Evaluation step outputs including loss, logits, etc.
        """
        with self.accelerator.autocast():
            outputs = self.forward(input_batch)

        loss = self.compute_loss(outputs, **input_batch)

        if self.model.is_generative and self.config.evaluate_with_generate:
            generate_inputs = sanitize_function_parameters(self.model.generate, input_batch)
            generated_ids = self.model.generate(**generate_inputs)
            outputs["logits"] = generated_ids["generated_ids"] if isinstance(generated_ids, dict) else generated_ids

        outputs["loss"] = loss

        return outputs

    def inner_training_loop(self, epoch_num: int):
        """
        Train the model for one epoch on the whole train dataset and verbose live metric values in the progress bar

        Args:
            epoch_num: Number of the current epoch

        Returns:
            Metrics averages through the full iteration
        """
        train_dataloader = self.create_train_dataloader(self.train_dataset)

        self.model.train()

        accumulated_loss = 0

        with tqdm(
            train_dataloader,
            initial=self.state.epoch_step,
            total=self.steps_in_epoch,
            unit="batch",
            desc=f"Epoch: {epoch_num}/{self.config.num_epochs} ",
            bar_format=TQDM_BAR_FORMAT,
            ascii=" #",
            disable=not self.accelerator.is_local_main_process,
        ) as iterator:
            for input_batch in iterator:
                # Handle early stopping
                if self.state.global_step >= self.total_steps:
                    break

                # Prepare inputs
                input_batch = self.prepare_input_batch(input_batch)

                # Training on one batch
                with self.accelerator.accumulate(self.model):
                    outputs = self.training_step(input_batch)
                    # Optimization step
                    self.optimization_step()

                # Update steps states
                self.state.global_step += 1
                self.state.epoch_step += 1

                # Gather outputs for metrics
                accumulated_loss += outputs["loss"].item()
                if (
                    self.state.epoch_step % self.config.gradient_accumulation_steps == 0
                    or self.state.epoch_step == self.steps_in_epoch
                ):
                    accumulated_loss = accumulated_loss / self.config.gradient_accumulation_steps
                    self.train_loss_tracker.update(accumulated_loss)
                    iterator.set_postfix(loss=self.train_loss_tracker.avg)

                    self.state.loss_tracker_avg = self.train_loss_tracker.avg
                    self.state.loss_tracker_sum = self.train_loss_tracker.sum
                    accumulated_loss = 0

                # Save trainer outputs if `save_steps` is hit
                if self.config.save_steps and self.state.global_step % self.config.save_steps == 0:
                    ckpt_path_name = str(self.state.global_step).zfill(len(str(self.total_steps)))
                    self.save(os.path.join(self.checkpoints_dir, ckpt_path_name))
                    # Save Trainer state
                    self.state.save(
                        os.path.join(
                            self.checkpoints_dir,
                            self.trainer_state_file,
                        )
                    )
                # Log loss running mean
                if self.config.log_steps and self.state.global_step % self.config.log_steps == 0:
                    loss_mean = {"train.loss": self.train_loss_tracker.avg}
                    write_to_tensorboard(self.tensorboard, logs=loss_mean, step=self.state.global_step)

        return {"loss": self.train_loss_tracker.avg}

    def evaluate(self, eval_dataset: Dataset = None):
        """
        Evaluates the model on the whole eval dataset and verbose live metric values in the progress bar

        Args:
            eval_dataset: Any sized iterable like a Hezar Dataset, HuggingFace Dataset, Torch Dataset, etc.

        Returns:
            A dictionary of evaluation results computed by the metrics tracker
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError(
                "Evaluation needs either passing the `eval_dataset` to the Trainer's `__init__` or `evaluate()`!"
            )
        eval_dataloader = self.create_eval_dataloader(eval_dataset or self.eval_dataset)

        self.metrics_handler.tracker.reset()
        self.model.eval()

        with tqdm(
            eval_dataloader,
            unit="batch",
            desc="Evaluating... ",
            bar_format=TQDM_BAR_FORMAT,
            ascii=" #",
            disable=not self.accelerator.is_local_main_process,
        ) as iterator:
            with torch.inference_mode():
                for step, input_batch in enumerate(iterator):
                    input_batch = self.prepare_input_batch(input_batch)
                    # Evaluation on one batch
                    outputs = self.evaluation_step(input_batch)
                    logits, labels = self.accelerator.gather_for_metrics((outputs["logits"], input_batch["labels"]))
                    # Compute metrics
                    evaluation_results = self.metrics_handler.compute_metrics(
                        logits.clone().detach().cpu(),
                        labels.clone().detach().cpu(),
                    )
                    evaluation_results["loss"] = self.accelerator.gather_for_metrics(outputs["loss"]).item()
                    # Gather outputs for metrics
                    self.metrics_handler.tracker.update(evaluation_results)
                    iterator.set_postfix(**self.metrics_handler.tracker.avg())

        return self.metrics_handler.tracker.avg()

    def print_info(self):
        """
        Print training info
        """

        def _print_info_line(key, value):
            line = f"  {colorize_text(key, 'bold')}: {colorize_text(str(value), 'italic')}"
            self.accelerator.print(line)

        header = f"{'*' * 20} Training Info {'*' * 20}"
        footer = "*" * len(header)
        info = {
            "Output Directory": self.config.output_dir,
            "Task": self.config.task,
            "Model": type(self.model).__name__,
            "Init Weights": self.config.init_weights_from or "N/A",
            "Device(s)": self.device,
            "Batch Size": self.config.batch_size,
            "Epochs": self.config.num_epochs,
            "Total Steps": self.total_steps,
            "Training Dataset": self.train_dataset,
            "Evaluation Dataset": self.eval_dataset,
            "Optimizer": self.config.optimizer or self.default_optimizer,
            "Scheduler": self.config.lr_scheduler,
            "Initial Learning Rate": self.config.learning_rate,
            "Learning Rate Decay": self.config.weight_decay,
            "Number of Parameters": self.model.num_parameters,
            "Number of Trainable Parameters": self.model.num_trainable_parameters,
            "Mixed Precision": self.config.mixed_precision or "Full (fp32)",
            "Gradient Accumulation Steps": self.config.gradient_accumulation_steps,
            "Metrics": list(self.metrics_handler.metrics.keys()),
            "Save Steps": self.config.save_steps,
            "Log Steps": self.config.log_steps,
            "Checkpoints Path": self.checkpoints_dir,
            "Logs Path": self.logs_dir,
        }

        # Header
        self.accelerator.print(f"\n{colorize_text(header, 'bold')}\n")
        # Info
        [_print_info_line(k, v) for k, v in info.items()]
        # Footer
        self.accelerator.print(f"\n{colorize_text(footer, 'bold')}\n")

    def train(self, resume_from_checkpoint: str | bool = None):
        """
        The full training process like training, evaluation, logging and saving model checkpoints.

        The steps are as follows:
            - The following is run for `self.config.num_epochs` times
                - Run the training loop on the train dataset
                - Save checkpoints
                - Run evaluation on the evaluation dataset
                - Apply LR scheduling of a LR Scheduler is available
                - Gather all metrics outputs
                - Save the trainer state
                - Write logs to tensorboard, csv, etc.
        """
        if resume_from_checkpoint:
            raise ValueError(
                "Setting `resume_from_checkpoint` in `Trainer.train(resume_from_checkpoint=...)` is deprecated. "
                "You have to set it in the trainer's config!"
            )

        self.print_info()

        for epoch in range(self.state.epoch, self.config.num_epochs + 1):
            self.accelerator.print()
            self.state.epoch = epoch
            metrics_logs = {}

            # Train on the whole training set
            training_results = self.inner_training_loop(epoch)
            metrics_logs.update({"train.loss": training_results["loss"]})

            # Save checkpoint
            if self.accelerator.is_local_main_process and self.config.save_enabled:
                ckpt_path_name = str(self.state.global_step).zfill(len(str(self.total_steps)))
                self.save(os.path.join(self.checkpoints_dir, ckpt_path_name))

            if self.config.do_evaluate:
                # Evaluate the model on the evaluation set
                evaluation_results = self.evaluate(self.eval_dataset)
                evaluation_logs = {
                    f"evaluation.{metric_name}": value for metric_name, value in evaluation_results.items()
                }
                metrics_logs.update(evaluation_logs)

            # LR scheduler step
            self.lr_scheduler_step(metrics_logs[self.config.metric_for_best_model])

            # Update trainer state
            self.state.epoch = epoch
            self.state.epoch_step = 0
            self.state.update_best_results(
                metric_value=metrics_logs[self.config.metric_for_best_model],
                objective=self.metrics_handler.objective,
                step=self.state.global_step,
            )

            # Log everything
            self.log(metrics_logs, self.state.global_step)

            # Early stopping
            if self.state.global_step == self.total_steps:
                break

        self.logger.info("Training done!")

    def log(self, logs: Dict[str, Any], step: int):
        """
        Log metrics results
        """
        # Log to tensorboard
        write_to_tensorboard(self.tensorboard, logs, step)

        # Log to CSV
        self.csv_logger.write(logs, step)

        # Save trainer state
        self.state.save(
            os.path.join(
                self.checkpoints_dir,
                self.trainer_state_file,
            )
        )

    def save(
        self,
        path: str,
        config_filename: str = None,
        model_filename: str = None,
        model_config_filename: str = None,
        subfolder: str = None,
        dataset_config_file: str = None,
        optimizer_file: str = None,
        lr_scheduler_file: str = None,
    ):
        """
        Save the trainer and relevant files to a path.

        Files to save are train config, model weights, model config, preprocessor files and preprocessor config.

        Args:
            path: A directory to save everything
            config_filename: Config file name
            model_filename: Model file name
            model_config_filename: Model config file name
            subfolder: Optional sub-folder
            dataset_config_file: Dataset config file name
            optimizer_file: Optimizer state file name
            lr_scheduler_file: LR scheduler file name
        """
        config_filename = config_filename or self.trainer_config_file
        subfolder = subfolder or self.trainer_subfolder
        dataset_config_file = dataset_config_file or self.dataset_config_file
        optimizer_file = optimizer_file or self.optimizer_file
        lr_scheduler_file = lr_scheduler_file or self.lr_scheduler_file

        self.config.save(path, filename=config_filename, subfolder=subfolder)
        self.model.save(path, filename=model_filename, config_filename=model_config_filename)

        torch.save(self.optimizer.state_dict(), os.path.join(path, subfolder, optimizer_file))

        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(path, subfolder, lr_scheduler_file))

        if isinstance(self.train_dataset, Dataset):
            self.train_dataset.config.save(path, filename=dataset_config_file, subfolder=subfolder)

    def push_to_hub(
        self,
        repo_id: str,
        config_filename: str = None,
        push_model: bool = True,
        push_optimizer: bool = True,
        push_logs: bool = True,
        model_filename: str = None,
        model_config_filename: str = None,
        optimizer_filename: str = None,
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
            push_model: Whether to push the model
            push_optimizer: Whether to push the optimizer
            push_logs: Whether to push training logs
            model_filename: Model file name
            optimizer_filename: Optimizer file name
            model_config_filename: Model config file name
            subfolder: Path to Trainer files
            dataset_config_filename: Dataset config file name
            commit_message: Commit message for the push
            private: Whether to create a private repo if it doesn't exist already
        """
        config_filename = config_filename or self.trainer_config_file
        subfolder = subfolder or self.trainer_subfolder
        dataset_config_file = dataset_config_filename or self.dataset_config_file
        optimizer_filename = optimizer_filename or self.optimizer_file

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

        # Upload train dataset config
        if isinstance(self.train_dataset, Dataset):
            self.train_dataset.config.push_to_hub(
                repo_id,
                filename=dataset_config_file,
                subfolder=subfolder,
                private=private,
                commit_message=commit_message
            )

        # Upload model
        if push_model:
            self.model.push_to_hub(
                repo_id,
                filename=model_filename,
                config_filename=model_config_filename,
                commit_message=commit_message,
                private=private,
            )

        # Upload optimizer state
        if push_optimizer:
            optimizer_path = os.path.join(tempfile.mkdtemp(), optimizer_filename)
            torch.save(self.optimizer.state_dict(), optimizer_path)
            upload_file(
                path_or_fileobj=optimizer_path,
                path_in_repo=os.path.join(self.trainer_subfolder, optimizer_filename),
                repo_id=repo_id,
                commit_message=commit_message,
            )
        # Upload logs
        if push_logs:
            upload_file(
                path_or_fileobj=self.csv_logger.save_path,
                path_in_repo=os.path.join(self.trainer_subfolder, self.trainer_csv_log_file),
                repo_id=repo_id,
                commit_message=commit_message,
            )
