from typing import List, Union

import numpy as np

from ..builders import build_metric
from ..configs import MetricConfig
from ..metrics import Metric
from ..models import Model
from ..constants import MetricType


class Evaluator:
    valid_metrics: List[MetricType]

    def __init__(self, metrics: List[Union[str, MetricType, Metric]], model_config=None, trainer_config=None):
        self.metrics = self._setup_metrics(metrics)
        self.model_config = model_config
        self.trainer_config = trainer_config

    def _setup_metrics(self, metrics):
        metrics_dict = {}
        for metric in metrics:
            if isinstance(metric, str):
                if metric not in self.valid_metrics:
                    raise ValueError(f"Invalid metric `{metric}`! Available metrics: {self.valid_metrics}")
                metrics_dict[metric] = build_metric(metric)
            elif isinstance(metric, MetricConfig):
                metrics_dict[metric.name] = build_metric(metric.name, config=metric)
            else:
                raise ValueError(f"Invalid metric type `{type(metric)}`! Available metrics: {self.valid_metrics}")
        return metrics_dict

    def compute_on_batch(self, predictions, labels, **kwargs):
        raise NotImplementedError


class TextClassificationEvaluator(Evaluator):
    valid_metrics = [
        MetricType.ACCURACY,
        MetricType.RECALL,
        MetricType.PRECISION,
        MetricType.F1,
    ]

    def __init__(self, metrics: List[Union[str, MetricType, Metric]], model_config=None, trainer_config=None):
        super().__init__(metrics=metrics, model_config=model_config, trainer_config=trainer_config)

    def compute_on_batch(self, predictions, labels, **kwargs):
        predictions = np.array(predictions).argmax(1).flatten()
        labels = np.array(labels).flatten()
        results = {}
        for metric_name, metric in self.metrics.items():
            results.update(metric.compute(predictions, labels))
        return results


class SequenceLabelingEvaluator(Evaluator):
    valid_metrics = [MetricType.SEQEVAL]

    def __init__(self, metrics: List[Union[str, MetricType, Metric]], model_config=None, trainer_config=None):
        super().__init__(metrics=metrics, model_config=model_config, trainer_config=trainer_config)

    def compute_on_batch(self, predictions, labels, **kwargs):
        predictions = np.array(predictions).argmax(2).squeeze()
        labels = np.array(labels).squeeze()

        # Remove ignored index (special tokens) and append `B-` in the beginning for seqeval
        prefix = "" if self.trainer_config.dataset.is_iob_schema else "B-"
        true_predictions = [
            [f"{prefix}{self.model_config.id2label[p]}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [f"{prefix}{self.model_config.id2label[l]}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = {}
        for metric_name, metric in self.metrics.items():
            x = metric.compute(true_predictions, true_labels)
            results.update(x)
        return results
