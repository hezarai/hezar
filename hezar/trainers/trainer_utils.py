from typing import Callable, Dict, Any

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


class MetricsManager:
    def __init__(self, metrics_dict: Dict[str, Callable]):
        self.metrics_dict = metrics_dict
        self.trackers = {m: AverageMeter(m) for m in self.metrics_dict.keys()}

    # def compute(self, preds, labels) -> Dict[str, Any]:
    #     results = {}
    #     for metric_name, metric_fn in self.metrics_dict.items():
    #         if metric_fn is not None:
    #             results[metric_name] = metric_fn(preds, labels).item()
    #
    #     return results

    def update(self, results):
        for metric_name, metric in self.trackers.items():
            metric.update(results[metric_name])

    def reset(self):
        for metric in self.trackers.values():
            metric.reset()

    def avg(self):
        avg_results = {}
        for metric_name, metric in self.trackers.items():
            avg_results[metric_name] = metric.avg

        return avg_results


def write_to_tensorboard(writer: SummaryWriter, logs: dict, mode: str, step: int):
    for metric_name, value in logs.items():
        writer.add_scalar(f"{mode}/{metric_name}", value, step)
