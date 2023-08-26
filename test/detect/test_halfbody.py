import pytest

from imgutils.detect.halfbody import _open_halfbody_detect_model, detect_halfbody
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_halfbody_detect_model.cache_clear()


@pytest.mark.unittest
class TestDetectHead:
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
