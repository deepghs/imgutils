import pytest

from imgutils.detect.person import _open_person_detect_model, detect_person
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_person_detect_model.cache_clear()


@pytest.mark.unittest
class TestDetectPerson:

    def test_detect_person(self):
        detections = detect_person(get_testfile('genshin_post.jpg'))
        assert len(detections) == 4

        values = []
        for bbox, label, score in detections:
            assert label == 'person'
            values.append((bbox, int(score * 1000) / 1000))

        assert values == pytest.approx([
            ((1, 141, 417, 720), 0.882),
            ((707, 5, 1264, 720), 0.83),
            ((617, 412, 801, 681), 0.773),
            ((376, 234, 558, 661), 0.75),
        ])

    def test_detect_person_none(self):
        assert detect_person(get_testfile('png_full.png')) == []
