import pytest

from imgutils.detect import detection_similarity
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
        detection = detect_halfbody(get_testfile('nude_girl.png'))
        similarity = detection_similarity(detection, [
            ((117, 0, 511, 484), 'halfbody', 0.8835344314575195)
        ])
        assert similarity >= 0.9

    def test_detect_halfbody_none(self):
        assert detect_halfbody(get_testfile('png_full.png')) == []
