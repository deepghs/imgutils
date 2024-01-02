import glob
import os.path

import pytest

from imgutils.validate import anime_style_age
from imgutils.validate.style_age import _open_anime_style_age_model, anime_style_age_score
from test.testings import get_testfile

_ROOT_DIR = get_testfile('anime_style_age')
_EXAMPLE_FILES = [
    (os.path.relpath(file, _ROOT_DIR), os.path.basename(os.path.dirname(file)))
    for file in glob.glob(get_testfile('anime_style_age', '**', '*.jpg'), recursive=True)
]


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_anime_style_age_model.cache_clear()


@pytest.mark.unittest
class TestValidatePortrait:
    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_anime_style_age(self, image, label):
        image_file = get_testfile('anime_style_age', image)
        tag, score = anime_style_age(image_file)
        assert tag == label

    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_anime_style_age_score(self, image, label):
        image_file = get_testfile('anime_style_age', image)
        scores = anime_style_age_score(image_file)
        assert scores[label] > 0.5
