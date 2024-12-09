import pytest

from imgutils.detect import detection_similarity
from imgutils.detect.eye import detect_eyes
from imgutils.generic.yolo import _open_models_for_repo_id
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestDetectEyes:
    def test_detect_eye(self):
        detection = detect_eyes(get_testfile('nude_girl.png'))
        similarity = detection_similarity(detection, [
            ((350, 159, 382, 173), 'eye', 0.7742469310760498),
            ((295, 169, 319, 181), 'eye', 0.7276312112808228)
        ])
        assert similarity >= 0.9

    def test_detect_eye_none(self):
        assert detect_eyes(get_testfile('png_full.png')) == []
