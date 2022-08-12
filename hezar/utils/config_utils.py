import logging


def merge_kwargs_into_config(config, args):
    for k, v in args:
        if hasattr(config, k):
            setattr(config, k, v)
        else:
            logging.warning(f'{str(config)} does not take `{k}` as a config parameter!')

    return config
