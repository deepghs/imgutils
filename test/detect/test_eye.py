import pytest

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
        detections = detect_eyes(get_testfile('nude_girl.png'))
        assert len(detections) == 2

        values = []
        for bbox, label, score in detections:
            assert label in {'eye'}
            values.append((bbox, int(score * 1000) / 1000))

        assert values == pytest.approx([
            ((350, 160, 382, 173), 0.788),
            ((294, 170, 319, 181), 0.756),
        ])

    def test_detect_eye_none(self):
        assert detect_eyes(get_testfile('png_full.png')) == []
