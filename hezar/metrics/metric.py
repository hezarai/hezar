from typing import Dict, List, Union

from ..configs import MetricConfig
from ..constants import Backends
from ..utils import verify_dependencies


class Metric:
    """
    The base metric class for all metrics in Hezar.

    Metrics are simple wrappers for casual ready-to-use metrics like in scikit-learn, etc. and it's strongly recommended
    not to reinvent the wheel. If a metric is already implemented by some package, use it! The only reason to implement
    such a module, is to make sure the metrics are treated the same all over the framework.
    """
    required_backends: Union[List[Union[str, Backends]]] = []

    def __init__(self, config: MetricConfig, **kwargs):
        # Check if all the required dependencies are installed
        verify_dependencies(self, self.required_backends)

        self.config = config.update(kwargs)

    def compute(self, predictions=None, targets=None, **kwargs) -> Dict:
        """
        Compute metric value for the given predictions against the targets
        Args:
            predictions: Prediction values
            targets: Ground truth values
            **kwargs: Extra arguments depending on the metric

        Returns:
            A dictionary of the results and scores
        """
        raise NotImplementedError
