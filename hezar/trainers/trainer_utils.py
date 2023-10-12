import os.path

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


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
        self.metrics = metrics
        self.trackers = {}
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
        logs = {k: [v] for k, v in logs.items()}
        row = pd.DataFrame(logs)
        row.name = step
        self.df = pd.concat([self.df, row])
        self.df.to_csv(self.save_path, index=False)


def write_to_tensorboard(writer: SummaryWriter, logs: dict, step: int):
    for metric_name, value in logs.items():
        writer.add_scalar(metric_name, value, step)
