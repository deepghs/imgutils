import pytest

from imgutils.detect.hand import _open_hand_detect_model, detect_hands
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_hand_detect_model.cache_clear()


@pytest.mark.unittest
class TestDetectHead:
    def test_detect_hands(self):
        detections = detect_hands(get_testfile('genshin_post.jpg'))
        assert len(detections) >= 4
