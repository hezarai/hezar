from __future__ import annotations

from typing import List

import numpy as np

from ..builders import build_metric
from ..configs import MetricConfig
from ..constants import MetricType
from ..metrics import Metric
from .trainer_utils import MetricsTracker


__all__ = [
    "MetricsHandler",
    "TextClassificationMetricsHandler",
    "SequenceLabelingMetricsHandler",
    "SpeechRecognitionMetricsHandler",
    "Image2TextMetricHandler",
    "TextGenerationMetricsHandler",
]


class MetricsHandler:
    """
    Base metrics handler class for computing metrics. Subclasses must implement `compute_metrics` method based on
    their specific task.

    Args:
        metrics: A list of metrics (metric raw name or Metric object)
        model_config: Optional model config
        trainer_config: Optional trainer config
    """
    valid_metrics: List[MetricType] = []

    def __init__(self, metrics: List[str | MetricType | Metric | MetricConfig], trainer=None, **kwargs):
        self.metrics = self._setup_metrics(metrics)
        self.trainer = trainer
        self.tracker = MetricsTracker(self.metrics)
        self.objective = self._configure_objective()

    def _configure_objective(self):
        """
        Figure out if the objective of the metric is to `minimize` or `maximize`
        """
        target_metric = self.trainer.config.metric_for_best_model
        objective_metric = target_metric.split(".")[1] if "." in target_metric else target_metric
        if "loss" in objective_metric:
            objective = "minimize"
        else:
            if objective_metric not in self.metrics:
                raise ValueError(
                    f"{objective_metric} is not a valid metric for this task, "
                    f"available metrics: {list(self.tracker.trackers.values())}"
                )
            objective = self.metrics[objective_metric].config.objective
        return objective

    def _setup_metrics(self, metrics):
        """
        Prepare a dictionary of metric names and their instances mappings.
        Args:
            metrics: A list of either names, configs or metric instances. Defaults to `self.valid_metrics`.

        Returns:

        """
        metrics_dict = {}
        metrics = metrics or []
        if not len(metrics):
            return metrics_dict
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

    def compute_metrics(self, predictions, labels, **kwargs):
        """
        Given a batch of predictions and a batch of labels, compute all metrics

        Args:
            predictions: Predictions batch usually containing logits
            labels: Ground truth labels batch
        """
        raise NotImplementedError


class TextClassificationMetricsHandler(MetricsHandler):
    valid_metrics = [
        MetricType.ACCURACY,
        MetricType.RECALL,
        MetricType.PRECISION,
        MetricType.F1,
    ]

    def __init__(self, metrics: List[str | MetricType | Metric | MetricConfig], trainer=None):
        super().__init__(metrics=metrics, trainer=trainer)

    def compute_metrics(self, predictions, labels, **kwargs):
        predictions = np.array(predictions).argmax(1).flatten()
        labels = np.array(labels).flatten()
        results = {}
        for metric_name, metric in self.metrics.items():
            results.update(metric.compute(predictions, labels))
        return results


class SequenceLabelingMetricsHandler(MetricsHandler):
    valid_metrics = [MetricType.SEQEVAL]

    def __init__(self, metrics: List[str | MetricType | Metric | MetricConfig], trainer=None):
        super().__init__(metrics=metrics, trainer=trainer)

    def compute_metrics(self, predictions, labels, **kwargs):
        predictions = np.array(predictions).argmax(2).squeeze()
        labels = np.array(labels).squeeze()

        # Remove ignored index (special tokens) and append `B-` in the beginning for seqeval
        prefix = "" if self.trainer.train_dataset.config.is_iob_schema else "B-"
        true_predictions = [
            [f"{prefix}{self.trainer.model.config.id2label[p]}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [f"{prefix}{self.trainer.model.config.id2label[l]}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = {}
        for metric_name, metric in self.metrics.items():
            x = metric.compute(true_predictions, true_labels)
            results.update(x)
        return results


class Image2TextMetricHandler(MetricsHandler):
    valid_metrics = [MetricType.CER, MetricType.WER]

    def __init__(self, metrics: List[str | MetricType | Metric | MetricConfig], trainer=None):
        super().__init__(metrics=metrics, trainer=trainer)

    def compute_metrics(self, predictions, labels, **kwargs):
        predictions = self.trainer.model.post_process(predictions)
        labels = self.trainer.model.post_process(labels)
        predictions = [x["text"] for x in predictions]
        labels = [x["text"] for x in labels]
        results = {}
        for metric_name, metric in self.metrics.items():
            x = metric.compute(predictions, labels)
            results.update(x)
        return results


class SpeechRecognitionMetricsHandler(MetricsHandler):
    valid_metrics = [MetricType.CER, MetricType.WER]

    def __init__(self, metrics: List[str | MetricType | Metric | MetricConfig], trainer=None):
        super().__init__(metrics=metrics, trainer=trainer)

    def compute_metrics(self, predictions, labels, **kwargs):
        predictions = self.trainer.model.post_process(predictions)
        labels = self.trainer.model.post_process(labels)
        predictions = [x["text"] for x in predictions]
        labels = [x["text"] for x in labels]
        results = {}
        for metric_name, metric in self.metrics.items():
            x = metric.compute(predictions, labels)
            results.update(x)
        return results


class TextGenerationMetricsHandler(MetricsHandler):
    valid_metrics = [MetricType.ROUGE, MetricType.BLEU]

    def __init__(self, metrics: List[str | MetricType | Metric | MetricConfig], trainer=None):
        super().__init__(metrics=metrics, trainer=trainer)

    def compute_metrics(self, predictions, labels, **kwargs):
        predictions = self.trainer.model.post_process(predictions)
        labels = self.trainer.model.post_process(labels)
        predictions = [x["text"] for x in predictions]
        labels = [x["text"] for x in labels]
        results = {}
        for metric_name, metric in self.metrics.items():
            x = metric.compute(predictions, labels)
            results.update(x)
        return results
