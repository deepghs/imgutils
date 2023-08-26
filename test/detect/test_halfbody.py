import pytest

from imgutils.detect.halfbody import _open_halfbody_detect_model, detect_halfbodies
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_halfbody_detect_model.cache_clear()


@pytest.mark.unittest
class TestDetectHead:
    def test_detect_halfbodies(self):
        detections = detect_halfbodies(get_testfile('nude_girl.png'))
        assert len(detections) == 1

        values = []
        for bbox, label, score in detections:
            assert label in {'halfbody'}
            values.append((bbox, int(score * 1000) / 1000))

        assert values == pytest.approx([
            ((106, 0, 512, 478), 0.916),
        ])

    def test_detect_halfbodies_none(self):
        assert detect_halfbodies(get_testfile('png_full.png')) == []
