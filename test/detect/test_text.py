import pytest

from imgutils.detect import detection_similarity
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

        assert detection_similarity(
            detections,
            [
                ((866, 45, 959, 69), 'text', 0.543),
                ((222, 68, 313, 102), 'text', 0.543),
                ((424, 82, 508, 113), 'text', 0.541),
                ((691, 101, 776, 129), 'text', 0.471)
            ],
        ) >= 0.9

    def test_detect_text_without_resize(self):
        detections = detect_text(get_testfile('ml2.jpg'), max_area_size=None)
        assert len(detections) == 9

        assert detection_similarity(
            detections,
            [
                ((360, 218, 474, 250), 'text', 0.686),
                ((119, 218, 203, 240), 'text', 0.653),
                ((392, 47, 466, 76), 'text', 0.617),
                ((593, 174, 666, 204), 'text', 0.616),
                ((179, 451, 672, 472), 'text', 0.591),
                ((633, 314, 747, 337), 'text', 0.59),
                ((392, 369, 517, 386), 'text', 0.589),
                ((621, 81, 681, 102), 'text', 0.566),
                ((209, 92, 281, 122), 'text', 0.423),
            ]
        ) >= 0.9

    def test_detect_text_none(self):
        assert detect_text(get_testfile('png_full.png')) == []
