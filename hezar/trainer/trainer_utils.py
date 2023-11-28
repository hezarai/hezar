import os
from dataclasses import dataclass, asdict

from omegaconf import OmegaConf
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

__all__ = [
    "AverageMeter",
    "MetricsTracker",
    "CSVLogger",
    "write_to_tensorboard",
]


@dataclass
class TrainerState:
    """
    A Trainer state is a container for holding specific updating values in the training process and is saved when
    checkpointing.

    Args:
        epoch: Current epoch, floating decimal represents the percentage of the current epoch completed
        total_epochs: Total epochs to train the model
        global_step: Number of the update steps so far, one step is a full training step (one batch)
        logging_steps: Log every X steps
        evaluation_steps: Evaluate every X steps
        best_checkpoint: Path to the best model checkpoint so far
        experiment_name: Name of the current run/experiment
    """
    epoch: float = 1.0
    total_epochs: int = None
    global_step: int = 0
    logging_steps: int = None
    evaluation_steps: int = None
    best_checkpoint: str = None
    experiment_name: str = None

    def update(self, items: dict, **kwargs):
        items.update(kwargs)
        for k, v in items.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def save(self, path, drop_none: bool = False):
        """
        Save the state to a .yaml file at `path`
        """
        state = asdict(self)
        if drop_none:
            state = {k: v for k, v in state.items() if v is not None}
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
