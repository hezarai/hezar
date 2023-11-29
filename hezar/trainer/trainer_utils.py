import os
from dataclasses import dataclass, asdict

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

__all__ = [
    "TrainerState",
    "AverageMeter",
    "MetricsTracker",
    "CSVLogger",
    "write_to_tensorboard",
    "resolve_logdir",
]


@dataclass
class TrainerState:
    """
    A Trainer state is a container for holding specific updating values in the training process and is saved when
    checkpointing.

    Args:
        epoch: Current epoch number
        total_epochs: Total epochs to train the model
        global_step: Number of the update steps so far, one step is a full training step (one batch)
        metric_for_best_checkpoint: The metric key for choosing the best checkpoint (Also given in the TrainerConfig)
        best_metric_value: The value of the best checkpoint saved so far
        best_checkpoint: Path to the best model checkpoint so far
    """
    epoch: int = 1
    total_epochs: int = None
    global_step: int = 0
    metric_for_best_checkpoint: str = None
    best_metric_value: float = None
    best_checkpoint: str = None

    def update(self, items: dict, **kwargs):
        items.update(kwargs)
        for k, v in items.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def update_best_results(self, metric_value, objective, step):
        if objective == "maximize":
            operator = np.greater
        elif objective == "minimize":
            operator = np.less
        else:
            raise ValueError(f"`objective` must be either `maximize` or `minimize`, got `{objective}`!")

        if self.best_metric_value is None:
            self.best_metric_value = metric_value
            self.best_checkpoint = step

        elif operator(metric_value, self.best_metric_value):
            self.best_metric_value = metric_value
            self.best_checkpoint = step

    def save(self, path, drop_none: bool = False):
        """
        Save the state to a .yaml file at `path`
        """
        state = asdict(self)
        if drop_none:
            state = {k: v for k, v in state.items() if v is not None}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        OmegaConf.save(state, path)

    @classmethod
    def load(cls, path):
        """
        Load a trainer state from `path`
        """
        state_file = OmegaConf.load(path)
        state_dict = OmegaConf.to_container(state_file)
        state = cls(**state_dict)
        return state


class AverageMeter:
    """Compute and store the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class MetricsTracker:
    def __init__(self, metrics):
        self.metrics = metrics or []
        self.trackers = {}
        if len(self.metrics):
            for m in self.metrics.values():
                for metric_key in m.config.output_keys:
                    self.trackers[metric_key] = AverageMeter(metric_key)
        if "loss" not in self.trackers:
            self.trackers["loss"] = AverageMeter("loss")

    def update(self, results):
        for metric_name, tracker in self.trackers.items():
            tracker.update(results[metric_name])

    def reset(self):
        for tracker in self.trackers.values():
            tracker.reset()

    def avg(self):
        avg_results = {}
        for metric_name, tracker in self.trackers.items():
            avg_results[metric_name] = tracker.avg

        return avg_results


class CSVLogger:
    def __init__(self, logs_dir: str, csv_filename: str):
        self.save_path = os.path.join(logs_dir, csv_filename)
        self.df = pd.DataFrame({})

    def write(self, logs: dict, step: int):
        all_logs = {"step": step}
        all_logs.update({k: [v] for k, v in logs.items()})
        row = pd.DataFrame(all_logs)
        self.df = pd.concat([self.df, row])
        self.df.to_csv(self.save_path, index=False)


def write_to_tensorboard(writer: SummaryWriter, logs: dict, step: int):
    for metric_name, value in logs.items():
        writer.add_scalar(metric_name, value, step)


def resolve_logdir(log_dir) -> str:
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join(log_dir, current_time + "_" + socket.gethostname())
