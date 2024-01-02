import pytest
from hbutils.testing import TextAligner

try:
    import torch
except (ImportError, ModuleNotFoundError):
    pass


@pytest.fixture(autouse=True)
def _try_import_torch():
    yield


@pytest.fixture()
def text_aligner():
    return TextAligner().multiple_lines()
