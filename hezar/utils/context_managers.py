from time import perf_counter


__all__ = [
    "exec_timer",
]


class exec_timer:
    """
    A context manager that captures the execution time of all the operations inside it

    Examples:
        >>> with exec_timer() as timer:
        >>>     # operations here
        >>> print(timer.time)
    """
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
