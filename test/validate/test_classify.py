import glob
import os.path

import pytest

from imgutils.generic.classify import _open_models_for_repo_id
from imgutils.validate import anime_classify
from imgutils.validate.classify import anime_classify_score, _REPO_ID
from test.testings import get_testfile

_ROOT_DIR = get_testfile('anime_cls')
_EXAMPLE_FILES = [
    (os.path.relpath(file, _ROOT_DIR), os.path.basename(os.path.dirname(file)))
    for file in glob.glob(get_testfile('anime_cls', '**', '*.jpg'), recursive=True)
]


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id(_REPO_ID).clear()


@pytest.mark.unittest
class TestValidateClassify:
    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_anime_classify(self, image, label):
        image_file = get_testfile('anime_cls', image)
        tag, score = anime_classify(image_file)
        assert tag == label

    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_anime_classify_score(self, image, label):
        image_file = get_testfile('anime_cls', image)
        scores = anime_classify_score(image_file)
        assert scores[label] > 0.5
