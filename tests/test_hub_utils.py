from typing import *

from omegaconf import DictConfig

from hezar.models import load_model, BaseModel


def test_push_to_hub():
    name = 'distilbert_text_classification'
    inner_model_config = DictConfig(
        {'pretrained_model_name_or_path': 'hezar-ai/distilbert-fa-zwnj-base', 'num_labels': 2})
    model: Type[BaseModel] = load_model(name, mode='training', inner_model_config=inner_model_config)
    model.push_to_hub(repo_name_or_id='hezar-ai/distilbert-fa-zwnj-base-test2')
    print()


if __name__ == '__main__':
    test_push_to_hub()
