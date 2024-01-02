import pytest

from imgutils.detect.text import _open_text_detect_model, detect_text
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_text_detect_model.cache_clear()


@pytest.mark.unittest
class TestDetectText:
    def test_detect_text(self):
        detections = detect_text(get_testfile('ml1.png'))
        assert len(detections) == 4

        values = []
        for bbox, label, score in detections:
            assert label in {'text'}
            values.append((bbox, int(score * 1000) / 1000))

        assert values == pytest.approx([
            ((866, 45, 959, 69), 0.543),
            ((222, 68, 313, 102), 0.543),
            ((424, 82, 508, 113), 0.541),
            ((691, 101, 776, 129), 0.471)
        ])

    def test_detect_text_without_resize(self):
        detections = detect_text(get_testfile('ml2.jpg'), max_area_size=None)
        assert len(detections) == 9

        values = []
        for bbox, label, score in detections:
            assert label in {'text'}
            values.append((bbox, int(score * 1000) / 1000))

        assert values == pytest.approx([
            ((360, 218, 474, 250), 0.686), ((119, 218, 203, 240), 0.653), ((392, 47, 466, 76), 0.617),
            ((593, 174, 666, 204), 0.616), ((179, 451, 672, 472), 0.591), ((633, 314, 747, 337), 0.59),
            ((392, 369, 517, 386), 0.589), ((621, 81, 681, 102), 0.566), ((209, 92, 281, 122), 0.423),
        ])

    def test_detect_text_none(self):
        assert detect_text(get_testfile('png_full.png')) == []
