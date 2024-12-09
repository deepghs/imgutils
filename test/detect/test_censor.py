import pytest

from imgutils.detect import detection_similarity
from imgutils.detect.censor import detect_censors
from imgutils.generic.yolo import _open_models_for_repo_id
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestDetectCensor:
    def test_detect_censors(self):
        detection = detect_censors(get_testfile('nude_girl.png'))
        similarity = detection_similarity(detection, [
            ((365, 264, 398, 289), 'nipple_f', 0.7295440435409546),
            ((207, 525, 237, 610), 'pussy', 0.7148708701133728),
            ((224, 261, 250, 287), 'nipple_f', 0.6702285408973694),
        ])
        assert similarity >= 0.9

    def test_detect_censors_none(self):
        assert detect_censors(get_testfile('png_full.png')) == []
