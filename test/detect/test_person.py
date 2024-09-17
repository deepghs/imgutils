import pytest

from imgutils.detect.person import detect_person
from imgutils.generic.yolo import _open_models_for_repo_id
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


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
            ((715, 8, 1268, 720), 0.861),
            ((57, 140, 407, 720), 0.856),
            ((614, 417, 802, 680), 0.802),
            ((373, 234, 561, 657), 0.782)
        ])

    def test_detect_person_none(self):
        assert detect_person(get_testfile('png_full.png'), conf_threshold=0.5) == []
