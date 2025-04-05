"""
This module provides a thread-safe version of Python's built-in lru_cache decorator.

The main component of this module is the ts_lru_cache decorator, which wraps the standard
lru_cache with thread-safe functionality. This is particularly useful in multi-threaded
environments where cache access needs to be synchronized to prevent race conditions.

Usage:
    >>> from imgutils.utils import ts_lru_cache
    ...
    >>> @ts_lru_cache(maxsize=100)
    >>> def expensive_function(x, y):
    ...     # Some expensive computation
    ...     return x + y
"""
import os
import threading
from collections import defaultdict
from functools import lru_cache, wraps

from typing import Literal

__all__ = ['ts_lru_cache']

LevelTyping = Literal['global', 'process', 'thread']


def _get_context_key(level: LevelTyping = 'global'):
    """
    Get a context key based on the specified caching level.

    :param level: The caching level to use. Can be 'global', 'process', or 'thread'.
    :type level: LevelTyping

    :return: A context key appropriate for the specified level.
    :rtype: tuple or None

    :raises ValueError: If an invalid cache level is specified.

    .. note::
        The function returns:

        - None for 'global' level
        - Process ID for 'process' level
        - (Process ID, Thread ID) tuple for 'thread' level
    """
    if level == 'global':
        return None
    elif level == 'process':
        return os.getpid()
    elif level == 'thread':
        return os.getpid(), threading.get_ident()
    else:
        raise ValueError(f'Invalid cache level, '
                         f'\'global\', \'process\' or \'thread\' expected but {level!r} found.')


def ts_lru_cache(level: LevelTyping = 'global', **options):
    """
    A thread-safe version of the lru_cache decorator.

    This decorator wraps the standard lru_cache with a threading lock to ensure
    thread-safety in multithreaded environments. It maintains the same interface
    as the built-in lru_cache, allowing you to specify options like maxsize.

    :param level: The caching level ('global', 'process', or 'thread').
    :type level: LevelTyping
    :param options: Keyword arguments to be passed to the underlying lru_cache.
    :type options: dict

    :return: A thread-safe cached version of the decorated function.
    :rtype: function

    :example:
        >>> @ts_lru_cache(level='thread', maxsize=100)
        >>> def my_function(x, y):
        ...     # Function implementation
        ...     return x + y

    .. note::
        The decorator provides three levels of caching:

        - global: Single cache shared across all processes and threads
        - process: Separate cache for each process
        - thread: Separate cache for each thread

    .. note::
        While this decorator ensures thread-safety, it may introduce some overhead
        due to lock acquisition. Use it when thread-safety is more critical than
        maximum performance in multithreaded scenarios.

    .. note::
        The decorator preserves the cache_info() and cache_clear() methods
        from the original lru_cache implementation.
    """
    _ = _get_context_key(level)

    def _decorator(func):
        """
        Inner decorator function that wraps the original function.

        :param func: The function to be decorated.
        :type func: function

        :return: The wrapped function with thread-safe caching.
        :rtype: function
        """

        @lru_cache(**options)
        @wraps(func)
        def _cached_func(*args, __context_key=None, **kwargs):
            """
            Cached version of the original function.

            :param args: Positional arguments to be passed to the original function.
            :param __context_key: Internal context key for cache separation.
            :param kwargs: Keyword arguments to be passed to the original function.

            :return: The result of the original function call.
            """
            return func(*args, **kwargs)

        lock_pool = defaultdict(threading.Lock)
        lock = threading.Lock()

        @wraps(_cached_func)
        def _new_func(*args, **kwargs):
            """
            Thread-safe wrapper around the cached function.

            This function acquires a lock before calling the cached function,
            ensuring thread-safety.

            :param args: Positional arguments to be passed to the cached function.
            :param kwargs: Keyword arguments to be passed to the cached function.

            :return: The result of the cached function call.
            """
            context_key = _get_context_key(level=level)
            with lock:
                _context_lock = lock_pool[context_key]
            with _context_lock:
                return _cached_func(*args, __context_key=context_key, **kwargs)

        # Preserve cache_info and cache_clear methods if they exist
        if hasattr(_cached_func, 'cache_info'):
            _new_func.cache_info = _cached_func.cache_info
        if hasattr(_cached_func, 'cache_clear'):
            _new_func.cache_clear = _cached_func.cache_clear

        return _new_func

    return _decorator
