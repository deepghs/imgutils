import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from imgutils.utils import ts_lru_cache


@pytest.fixture
def slow_function():
    call_count = 0

    @ts_lru_cache(maxsize=None)
    def func(x):
        nonlocal call_count
        call_count += 1
        time.sleep(0.1)
        return x * 2

    return func, lambda: call_count


@pytest.fixture
def threaded_calls():
    def run_in_threads(func, args_list, num_threads):
        tp = ThreadPoolExecutor(max_workers=num_threads)
        futures = []
        for arg in args_list:
            futures.append(tp.submit(func, *arg))
        results = [f.result() for f in futures]
        tp.shutdown(wait=True)
        return results

    return run_in_threads


@pytest.mark.unittest
class TestTsLruCache:

    def test_single_thread_caching(self, slow_function):
        func, get_call_count = slow_function

        assert func(2) == 4
        assert func(2) == 4
        assert get_call_count() == 1

        assert func(3) == 6
        assert get_call_count() == 2

    def test_multi_thread_caching(self, slow_function, threaded_calls):
        func, get_call_count = slow_function

        args_list = [(2,), (2,), (3,), (3,)]
        results = threaded_calls(func, args_list, num_threads=4)

        assert results == [4, 4, 6, 6]
        assert get_call_count() == 2

    @pytest.mark.parametrize(['threads', 'total'], [(i, max(i * 5, i * i)) for i in range(2, 11)])
    def test_diff_threads(self, threads, total, slow_function, threaded_calls):
        func, get_call_count = slow_function

        args_list = [(i % threads,) for i in range(total)]
        results = threaded_calls(func, args_list, num_threads=threads)

        assert results == [(i % threads) * 2 for i in range(total)]
        assert get_call_count() == threads

    def test_different_args(self, slow_function, threaded_calls):
        func, get_call_count = slow_function

        args_list = [(i,) for i in range(10)]
        results = threaded_calls(func, args_list, num_threads=5)

        assert results == [i * 2 for i in range(10)]
        assert get_call_count() == 10

    def test_with_kwargs(self):
        call_count = 0

        @ts_lru_cache(maxsize=None)
        def func(x, y=1):
            nonlocal call_count
            call_count += 1
            return x * y

        assert func(2) == 2
        assert func(2) == 2
        assert func(2) == 2
        assert func(2, y=2) == 4
        assert func(2) == 2
        assert call_count == 2

    def test_without_args(self):
        call_count = 0

        @ts_lru_cache(maxsize=None)
        def func():
            nonlocal call_count
            call_count += 1
            return "result"

        assert func() == "result"
        assert func() == "result"
        assert call_count == 1

    def test_with_maxsize(self):
        call_count = 0

        @ts_lru_cache(maxsize=2)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert func(1) == 2
        assert func(2) == 4
        assert func(3) == 6  # This should evict func(1) from cache
        assert func(2) == 4  # This should hit func(2)
        assert func(1) == 2  # This should evict func(3) call_count
        assert func(4) == 8  # This should evict func(2) from cache

        assert call_count == 5

        assert func(2) == 4  # This should calculate again
        assert call_count == 6
