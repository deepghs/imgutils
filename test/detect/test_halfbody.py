import pytest

from imgutils.detect.halfbody import detect_halfbody
from imgutils.generic.yolo import _open_models_for_repo_id
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestDetectHalfBody:
    def test_detect_halfbody(self):
        detections = detect_halfbody(get_testfile('nude_girl.png'))
        assert len(detections) == 1

        values = []
        for bbox, label, score in detections:
            assert label in {'halfbody'}
            values.append((bbox, int(score * 1000) / 1000))

        assert values == pytest.approx([
            ((105, 0, 511, 480), 0.918),
        ])

    def test_detect_halfbody_none(self):
        assert detect_halfbody(get_testfile('png_full.png')) == []
