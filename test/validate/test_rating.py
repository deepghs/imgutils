import glob
import os.path

import pytest

from imgutils.validate import anime_rating
from imgutils.validate.rating import _open_anime_rating_model, anime_rating_score
from test.testings import get_testfile

_ROOT_DIR = get_testfile('rating')
_EXAMPLE_FILES = [
    (os.path.relpath(file, _ROOT_DIR), os.path.basename(os.path.dirname(file)))
    for file in glob.glob(get_testfile('rating', '**', '*.jpg'), recursive=True)
]


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_anime_rating_model.cache_clear()


@pytest.mark.unittest
class TestValidateRating:
    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_anime_rating(self, image, label):
        image_file = get_testfile('rating', image)
        tag, score = anime_rating(image_file)
        assert tag == label

    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_anime_rating_score(self, image, label):
        image_file = get_testfile('rating', image)
        scores = anime_rating_score(image_file)
        assert scores[label] > 0.5
