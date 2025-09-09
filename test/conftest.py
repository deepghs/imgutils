import os

import pytest
from hbutils.testing import TextAligner
from hfutils.cache import delete_cache

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


@pytest.fixture(autouse=True, scope='module')
def clean_hf_cache():
    if os.environ.get('CI'):
        delete_cache()
