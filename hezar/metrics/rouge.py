from dataclasses import dataclass

from .metric import Metric
from ..configs import MetricConfig
from ..constants import Backends, MetricType
from ..registry import register_metric
from ..utils import is_backend_available

if is_backend_available(Backends.EVALUATE):
    import evaluate

_DESCRIPTION = "Rouge estimation using `evaluate`. Commonly used for Text Summarization"

_required_backends = [
    Backends.EVALUATE,
]


@dataclass
class ROUGEConfig(MetricConfig):
    name = MetricType.ROUGE
    use_stemmer: bool = True
    output_keys: tuple = ("rouge",)


@register_metric("rouge", config_class=ROUGEConfig, description=_DESCRIPTION)
class ROUGE(Metric):
    required_backends = _required_backends

    def __init__(self, config: ROUGEConfig, **kwargs):
        super().__init__(config=config, **kwargs)

    def compute(
        self,
        predictions=None,
        targets=None,
        concatenate_texts=None,
        n_decimals=None,
        output_keys=None,
        **kwargs,
    ):
        rouge = evaluate.load("rouge")

        results = rouge.compute(predictions=predictions, references=targets, use_stemmer=self.config.use_stemmer)

        if output_keys:
            results = {k: round(v, 4) for k, v in results.items() if k in output_keys}

        return results
