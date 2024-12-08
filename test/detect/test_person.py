import pytest

from imgutils.detect import detection_similarity
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
        similarity = detection_similarity(detections, [
            ((35, 143, 398, 720), 'person', 0.8631670475006104),
            ((690, 1, 1262, 712), 'person', 0.845992922782898),
            ((373, 236, 561, 638), 'person', 0.7979342937469482),
            ((607, 418, 804, 680), 'person', 0.7805007696151733)
        ])
        assert similarity >= 0.85

    def test_detect_person_none(self):
        assert detect_person(get_testfile('png_full.png'), conf_threshold=0.5) == []
