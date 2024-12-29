import os
import time
from typing import Dict, List
from contextlib import contextmanager

import pytest
import logging
from hezar.builders import build_model
from hezar.models import ModelConfig
from hezar.preprocessors import Preprocessor
from hezar.utils import clean_cache, set_seed

logger = logging.getLogger(__name__)

CI_MODE = os.environ.get("CI_MODE", "FALSE")

TESTABLE_MODELS: Dict[str, Dict] = {
    "speech-recognition": {
        "path": "hezarai/whisper-small-fa",
        "inputs": {"type": "file", "value": "samples/speech_example.mp3"},
        "predict_kwargs": {},
        "output_type_within_batch": dict,
        "required_output_keys": {"text", "chunks"}
    },
    # ... other models
}

INVALID_OUTPUT_TYPE = "Model output must be a batch!"
INVALID_OUTPUT_SIZE = "Model output must be a list of size 1!"
INVALID_OUTPUT_FIELDS = "Invalid fields in the model outputs!"

@contextmanager
def time_context(task: str):
    """Context manager to measure the execution time of a test."""
    start_time = time.time()
    yield
    end_time = time.time()
    logger.info(f"Test for '{task}' completed in {end_time - start_time:.2f} seconds.")

@pytest.mark.parametrize("task", TESTABLE_MODELS.keys())
def test_model_inference(task):
    set_seed(42)

    model_params = TESTABLE_MODELS[task]
    path = model_params["path"]
    predict_kwargs = model_params["predict_kwargs"]
    output_type_within_batch = model_params["output_type_within_batch"]
    required_output_keys = model_params["required_output_keys"]

    if model_params["inputs"]["type"] == "file":
        dirname = os.path.dirname(os.path.abspath(__file__))
        inputs = os.path.join(dirname, model_params["inputs"]["value"])
    else:
        inputs = model_params["inputs"]["value"]

    model_config = ModelConfig.load(path)
    model = build_model(model_config.name, config=model_config)
    model.preprocessor = Preprocessor.load(path)

    with time_context(task):
        outputs = model.predict(inputs, **predict_kwargs)

    assert isinstance(outputs, list), INVALID_OUTPUT_TYPE
    assert len(outputs) == 1, INVALID_OUTPUT_SIZE
    if output_type_within_batch == list:
        assert {k for el in outputs[0] for k in el.keys()} == required_output_keys
    elif output_type_within_batch == dict:
        assert set(outputs[0].keys()) == required_output_keys, INVALID_OUTPUT_FIELDS

    if CI_MODE == "TRUE":
        clean_cache(delay=1)

def test_report_generation():
    """Test the reporting and visualization features."""
    report = generate_test_report(TESTABLE_MODELS.keys())
    assert report is not None
    assert "success_rate" in report
    assert "failure_rate" in report
    assert "average_inference_time" in report

    visualize_test_results(report)
    assert os.path.exists("test_results.html")

def generate_test_report(tasks: List[str]) -> Dict:
    """Generate a test report for the given tasks."""
    report = {
        "success_rate": 0.0,
        "failure_rate": 0.0,
        "average_inference_time": 0.0
    }

    total_time = 0.0
    total_tests = 0
    successful_tests = 0

    for task in tasks:
        try:
            with time_context(task):
                test_model_inference(task)
            successful_tests += 1
        except AssertionError as e:
            logger.error(f"Test for '{task}' failed: {e}")
        finally:
            total_tests += 1
            total_time += time_context.duration

    report["success_rate"] = successful_tests / total_tests
    report["failure_rate"] = 1.0 - report["success_rate"]
    report["average_inference_time"] = total_time / total_tests

    return report

def visualize_test_results(report: Dict):
    """Generate a HTML report with the test results."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("test_report.html")
    html = template.render(report=report)

    with open("test_results.html", "w") as f:
        f.write(html)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main()
