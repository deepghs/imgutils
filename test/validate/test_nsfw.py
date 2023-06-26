import glob
import os.path

import pytest

from imgutils.validate import nsfw_pred
from imgutils.validate.nsfw import _open_nsfw_model, nsfw_pred_score
from test.testings import get_testfile

_ROOT_DIR = get_testfile('nsfw')
_EXAMPLE_FILES = [
    (os.path.relpath(file, _ROOT_DIR), os.path.basename(os.path.dirname(file)))
    for file in glob.glob(get_testfile('nsfw', '**', '*.jpg'), recursive=True)
]


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_nsfw_model.cache_clear()


@pytest.mark.unittest
class TestValidateNSFW:
    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_nsfw_pred(self, image, label):
        image_file = get_testfile('nsfw', image)
        tag, score = nsfw_pred(image_file)
        assert tag == label

    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_nsfw_pred_score(self, image, label):
        image_file = get_testfile('nsfw', image)
        scores = nsfw_pred_score(image_file)
        assert scores[label] > 0.5
