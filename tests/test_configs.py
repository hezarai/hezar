from hezar.configs import ModelConfig


def test_load_config(path):
    pretrained_config = ModelConfig.load(path)
    return pretrained_config


if __name__ == '__main__':
    config = test_load_config('test')
    print(config)
