from dataclasses import dataclass
from typing import List, Optional, Union

from seqeval.metrics import accuracy_score, classification_report

from ..configs import MetricConfig
from ..constants import MetricType
from ..registry import register_metric
from ..utils import Logger
from .metric import Metric


logger = Logger(__name__)


@dataclass
class SeqevalConfig(MetricConfig):
    name = MetricType.SEQEVAL
    suffix: bool = False
    mode: Optional[str] = None
    sample_weight: Optional[List[int]] = None
    zero_division: Union[str, int] = 0
    digits: int = None


@register_metric("seqeval", config_class=SeqevalConfig)
class Seqeval(Metric):
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
        digits: int = None,
        **kwargs,
    ):
        suffix = suffix or self.config.suffix
        mode = mode or self.config.mode
        sample_weight = sample_weight or self.config.sample_weight
        zero_division = zero_division or self.config.zero_division
        digits = digits or self.config.digits

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
        if digits:
            results = {k: round(float(v), digits) for k, v in results.items()}
        return results



