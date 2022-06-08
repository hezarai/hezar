from hezar.configs import ModelConfig, TrainerConfig, DatasetConfig
from hezar.models import RobertaTextClassificationConfig


def test_load_config(path, config_class):
    raw_config = ModelConfig(config_class=config_class)
    pretrained_config = raw_config.from_pretrained(path)
    return pretrained_config


if __name__ == '__main__':
    config = test_load_config('assets/config.yaml', config_class=None)
    print(config)
