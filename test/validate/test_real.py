import glob
import os.path

import pytest

from imgutils.validate import anime_real
from imgutils.validate.real import _open_anime_real_model, anime_real_score
from test.testings import get_testfile

_ROOT_DIR = get_testfile('real')
_EXAMPLE_FILES = [
    (os.path.relpath(file, _ROOT_DIR), os.path.basename(os.path.dirname(file)))
    for file in glob.glob(get_testfile('real', '**', '*.jpg'), recursive=True)
]


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_anime_real_model.cache_clear()


@pytest.mark.unittest
class TestValidatePortrait:
    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_anime_real(self, image, label):
        image_file = get_testfile('real', image)
        tag, score = anime_real(image_file)
        assert tag == label

    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_anime_real_score(self, image, label):
        image_file = get_testfile('real', image)
        scores = anime_real_score(image_file)
        assert scores[label] > 0.5
