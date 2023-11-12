from dataclasses import dataclass, field

from .metric import Metric
from ..configs import MetricConfig
from ..constants import Backends, MetricType
from ..registry import register_metric
from ..utils import is_backend_available

if is_backend_available(Backends.ROUGE):
    from rouge_score import rouge_scorer, scoring

_DESCRIPTION = "Rouge estimation. Commonly used for Text Summarization"

_required_backends = [
    Backends.ROUGE,
]


@dataclass
class ROUGEConfig(MetricConfig):
    name = MetricType.ROUGE
    use_stemmer: bool = False
    use_aggregator: bool = True
    multi_ref: bool = True
    output_keys: tuple = ("rouge1", "rouge2", "rougeL", "rougeLsum",)


@register_metric("rouge", config_class=ROUGEConfig, description=_DESCRIPTION)
class ROUGE(Metric):
    required_backends = _required_backends

    def __init__(self, config: ROUGEConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=rouge_types,
            use_stemmer=self.config.use_stemmer,
        )

    def compute(
        self,
        predictions=None,
        targets=None,
        use_aggregator=None,
        n_decimals=None,
        output_keys=None,
        **kwargs,
    ):
        aggregator = scoring.BootstrapAggregator()

        for ref, pred in zip(targets, predictions):
            if self.config.multi_ref:
                score = self.scorer.score_multi(ref, pred)
            else:
                score = self.scorer.score(ref, pred)

            aggregator.add_scores(score)

        results = aggregator.aggregate()
        for key in results:
            results[key] = results[key].mid.fmeasure

        if output_keys:
            results = {k: round(v, 4) for k, v in results.items() if k in output_keys}

        return results
