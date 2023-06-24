import glob
import os.path

import pytest

from imgutils.validate.aicheck import _open_anime_aicheck_model, is_ai_created, get_ai_created_score
from test.testings import get_testfile

_ROOT_DIR = get_testfile('anime_aicheck')
_EXAMPLE_FILES = [
    (os.path.relpath(file, _ROOT_DIR), os.path.basename(os.path.dirname(file)))
    for file in glob.glob(get_testfile('anime_aicheck', '**', '*.jpg'), recursive=True)
]


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_anime_aicheck_model.cache_clear()


@pytest.mark.unittest
class TestValidateAicheck:
    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_is_ai_created(self, image, label):
        image_file = get_testfile('anime_aicheck', image)
        is_ai = is_ai_created(image_file)
        if label == 'ai':
            assert is_ai, f'Label: {label!r}, predict: {is_ai!r}'
        else:
            assert not is_ai, f'Label: {label!r}, predict: {is_ai!r}'

    @pytest.mark.parametrize(['image', 'label'], _EXAMPLE_FILES)
    def test_get_ai_created_score(self, image, label):
        image_file = get_testfile('anime_aicheck', image)
        score = get_ai_created_score(image_file)
        if label == 'ai':
            assert score >= 0.5, f'Label: {label!r}, score: {score!r}'
        else:
            assert score < 0.5, f'Label: {label!r}, score: {score!r}'
