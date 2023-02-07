from .logging import get_logger

logger = get_logger(__name__)


def merge_kwargs_into_config(config, args):
    for k, v in args.items():
        if hasattr(config, k):
            setattr(config, k, v)
        else:
            logger.warning(f'{str(config.__class__.__name__)} does not take `{k}` as a config parameter!')

    return config
