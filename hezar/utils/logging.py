import logging


__all__ = [
    "get_logger",
]


def get_logger(name, level=None, fmt=None):
    fmt = fmt or "%(levelname)s: %(message)s"
    level = level or "INFO"

    logger = logging.Logger(name, level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
