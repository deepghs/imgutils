import glob
import os.path

import pytest

from imgutils.validate import anime_bangumi_char
from imgutils.validate.bangumi_char import _open_anime_bangumi_char_model, anime_bangumi_char_score
from test.testings import get_testfile

_ROOT_DIR = get_testfile('bangumi_char')
_EXAMPLE_FILES = [
    (os.path.relpath(file, _ROOT_DIR), os.path.basename(os.path.dirname(file)))
    for file in glob.glob(get_testfile('bangumi_char', '**', '*.jpg'), recursive=True)
]


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_anime_bangumi_char_model.cache_clear()


@pytest.mark.unittest
class TestValidateBangumiChar:
    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_anime_bangumi_char(self, image, label):
        image_file = get_testfile('bangumi_char', image)
        tag, score = anime_bangumi_char(image_file)
        assert tag == label

    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_anime_bangumi_char_score(self, image, label):
        image_file = get_testfile('bangumi_char', image)
        scores = anime_bangumi_char_score(image_file)
        assert scores[label] > 0.5
