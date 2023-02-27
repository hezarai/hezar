import torch
from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def compute_metrics(metrics: dict, preds: torch.Tensor, labels: torch.Tensor):
    results = {}
    for metric_name, metric in metrics.items():
        results[metric_name] = metric(preds, labels).item()

    return results


def reset_trackers(trackers: dict):
    for metric in trackers.values():
        metric.reset()


def update_trackers(trackers: dict, metrics_results: dict):
    for metric_name, metric in trackers.items():
        metric.update(metrics_results[metric_name])


def get_trackers_avg(trackers: dict):
    avg_results = {}
    for metric_name, metric in trackers.items():
        avg_results[metric_name] = metric.avg

    return avg_results


def write_to_tensorboard(writer: SummaryWriter, logs: dict, mode: str, step: int):
    for metric_name, value in logs.items():
        writer.add_scalar(f"{mode}/{metric_name}", value, step)
