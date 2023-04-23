import pytest

try:
    import torch
except (ImportError, ModuleNotFoundError):
    pass


@pytest.fixture(autouse=True)
def _try_import_torch():
    yield
