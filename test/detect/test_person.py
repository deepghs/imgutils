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
            ((371, 232, 564, 690), 0.753),
            ((30, 135, 451, 716), 0.678),
            ((614, 393, 830, 686), 0.561),
            ((614, 3, 1275, 654), 0.404)
        ])

    def test_detect_person_none(self):
        assert detect_person(get_testfile('png_full.png')) == []
