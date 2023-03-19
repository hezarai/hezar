from time import perf_counter


__all__ = [
    "exec_timer",
]


class exec_timer:
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
