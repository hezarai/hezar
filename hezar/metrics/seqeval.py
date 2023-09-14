from dataclasses import dataclass
from typing import List, Optional, Union

from ..configs import MetricConfig
from ..constants import Backends, MetricType
from ..registry import register_metric
from ..utils import Logger, is_backend_available
from .metric import Metric


if is_backend_available(Backends.SEQEVAL):
    from seqeval.metrics import accuracy_score, classification_report

_required_backends = [
    Backends.SEQEVAL,
]

logger = Logger(__name__)


@dataclass
class SeqevalConfig(MetricConfig):
    name = MetricType.SEQEVAL
    output_keys: tuple = ("accuracy", "recall", "precision", "f1")
    suffix: bool = False
    mode: Optional[str] = None
    sample_weight: Optional[List[int]] = None
    zero_division: Union[str, int] = 0


@register_metric("seqeval", config_class=SeqevalConfig)
class Seqeval(Metric):
    required_backends = _required_backends

    def __init__(self, config: SeqevalConfig, **kwargs):
        super().__init__(config, **kwargs)

    def compute(
        self,
        predictions=None,
        targets=None,
        suffix: bool = None,
        mode: Optional[str] = None,
        sample_weight: Optional[List[int]] = None,
        zero_division: Union[str, int] = None,
        n_decimals: int = None,
        output_keys=None,
        **kwargs,
    ):
        suffix = suffix or self.config.suffix
        mode = mode or self.config.mode
        sample_weight = sample_weight or self.config.sample_weight
        zero_division = zero_division or self.config.zero_division
        n_decimals = n_decimals or self.config.n_decimals
        output_keys = output_keys or self.config.output_keys

        report = classification_report(
            y_true=targets,
            y_pred=predictions,
            suffix=suffix,
            output_dict=True,
            mode=mode,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        report.pop("macro avg")
        report.pop("weighted avg")
        overall_score = report.pop("micro avg")

        results = {
            "accuracy": format(accuracy_score(predictions, targets)),
            "f1": overall_score["f1-score"],
            "recall": overall_score["recall"],
            "precision": overall_score["precision"],
        }

        results = {k: round(float(v), n_decimals) for k, v in results.items()}

        if output_keys:
            results = {k: v for k, v in results.items() if k in output_keys}

        return results


