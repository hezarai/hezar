import logging


__all__ = [
    "Logger"
]


class Logger(logging.Logger):
    def __init__(self, name: str, level=None, fmt=None):
        fmt = fmt or "%(levelname)s: %(message)s"
        level = level or "INFO"
        super().__init__(name, level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        self.addHandler(handler)

    def log_upload_success(self, module, path_in_repo: str):
        src = f"{module.__class__.__name__}(name={module.config.name})"
        self.info(f"Uploaded: `{src}` --> `{path_in_repo}`")
