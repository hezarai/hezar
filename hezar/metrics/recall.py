from typing import Any, Dict

from sklearn.metrics import recall_score

from .metric import Metric
from ..registry import register_metric


@register_metric("recall")
class Recall(Metric):
    def compute(
        self,
        predictions=None,
        targets=None,
        labels=None,
        pos_label=1,
        average="binary",
        sample_weight=None,
        zero_division="warn",
    ) -> Dict[str, Any]:

        score = recall_score(
            targets,
            predictions,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

        return {"recall": float(score) if score.size == 1 else score}
