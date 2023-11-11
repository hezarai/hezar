from dataclasses import dataclass

from .metric import Metric
from ..configs import MetricConfig
from ..constants import Backends, MetricType
from ..registry import register_metric
from ..utils import is_backend_available

if is_backend_available(Backends.ROUGE):
    import rouge

_DESCRIPTION = "Rouge estimation. Commonly used for Text Summarization"

_required_backends = [
    Backends.ROUGE,
]


@dataclass
class ROUGEConfig(MetricConfig):
    name = MetricType.ROUGE

    output_keys: tuple = ("rouge",)


@register_metric("rouge", config_class=ROUGEConfig, description=_DESCRIPTION)
class ROUGE(Metric):
    required_backends = _required_backends

    def __init__(self, config: ROUGEConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.rouge = rouge.Rouge()

    def compute(
        self,
        predictions=None,
        targets=None,
        concatenate_texts=None,
        n_decimals=None,
        output_keys=None,
        **kwargs,
    ):
        results = self.rouge.get_scores(predictions, targets)
        output_result = {}
        for rouge_key, rouge_metrics in results[0].items():
            for metric_name, metric_value in rouge_metrics.items():
                output_result[f"{rouge_key}-{metric_name}"] = metric_value

        if output_keys:
            output_result = {k: round(v, 4) for k, v in output_result.items() if k in output_keys}

        return output_result
