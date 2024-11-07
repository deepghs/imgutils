import threading
from functools import lru_cache

__all__ = ['ts_lru_cache']


def ts_lru_cache(**options):
    def _decorator(func):
        @lru_cache(**options)
        def _cached_func(*args, **kwargs):
            return func(*args, **kwargs)

        lock = threading.Lock()

        def _new_func(*args, **kwargs):
            with lock:
                return _cached_func(*args, **kwargs)

        return _new_func

    return _decorator
