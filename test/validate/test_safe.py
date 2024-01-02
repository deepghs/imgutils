import glob
import os.path

import pytest

from imgutils.validate import safe_check, safe_check_score
from imgutils.validate.safe import _open_model
from test.testings import get_testfile

_ROOT_DIR = get_testfile('safe_check')
_EXAMPLE_FILES = [
    (os.path.relpath(file, _ROOT_DIR), os.path.basename(os.path.dirname(file)))
    for file in glob.glob(get_testfile('safe_check', '**', '*.webp'), recursive=True)
]


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_model.cache_clear()


@pytest.mark.unittest
class TestValidateSafe:
    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_safe_check(self, image, label):
        image_file = get_testfile('safe_check', image)
        tag, score = safe_check(image_file)
        assert tag == label

    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_safe_check_score(self, image, label):
        image_file = get_testfile('safe_check', image)
        scores = safe_check_score(image_file)
        assert scores[label] > 0.5
