import logging


__all__ = [
    "Logger"
]


class Logger(logging.Logger):
    def __init__(self, name: str, level=None, fmt=None):
        fmt = fmt or "Hezar (%(levelname)s): %(message)s"
        level = level or "INFO"
        super().__init__(name, level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        self.addHandler(handler)

    def log_upload_success(self, name, target_path: str):
        """
        Log (info) success info when the file(s) upload is done.
        """
        self.info(f"Uploaded: `{name}` --> `{target_path}`")
