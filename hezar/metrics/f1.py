from typing import Any, Dict

from sklearn.metrics import f1_score

from .metric import Metric
from ..registry import register_metric


@register_metric("f1")
class F1(Metric):
    def compute(
        self,
        predictions=None,
        targets=None,
        labels=None,
        pos_label=1,
        average="binary",
        sample_weight=None
    ) -> Dict[str, Any]:

        score = f1_score(
            targets,
            predictions,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
        )

        return {"f1": float(score) if score.size == 1 else score}
