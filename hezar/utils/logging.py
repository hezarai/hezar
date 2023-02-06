import logging


def get_logger(name, level=None, fmt=None):
    fmt = fmt or '%(name)s - %(funcName)s - %(levelname)s - %(message)s'
    level = level or 'INFO'

    logger = logging.Logger(name, level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
