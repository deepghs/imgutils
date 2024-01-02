import glob
import os.path

import pytest

from imgutils.validate import anime_portrait
from imgutils.validate.portrait import _open_anime_portrait_model, anime_portrait_score
from test.testings import get_testfile

_ROOT_DIR = get_testfile('anime_portrait')
_EXAMPLE_FILES = [
    (os.path.relpath(file, _ROOT_DIR), os.path.basename(os.path.dirname(file)))
    for file in glob.glob(get_testfile('anime_portrait', '**', '*.jpg'), recursive=True)
]


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_anime_portrait_model.cache_clear()


@pytest.mark.unittest
class TestValidatePortrait:
    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_anime_portrait(self, image, label):
        image_file = get_testfile('anime_portrait', image)
        tag, score = anime_portrait(image_file)
        assert tag == label

    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_anime_portrait_score(self, image, label):
        image_file = get_testfile('anime_portrait', image)
        scores = anime_portrait_score(image_file)
        assert scores[label] > 0.5
