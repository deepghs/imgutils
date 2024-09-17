import pytest

from imgutils.detect.face import detect_faces
from imgutils.generic.yolo import _open_models_for_repo_id
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestDetectHead:
    def test_detect_faces(self):
        detections = detect_faces(get_testfile('genshin_post.jpg'))
        assert len(detections) == 4

        values = []
        for bbox, label, score in detections:
            assert label == 'face'
            values.append((bbox, int(score * 1000) / 1000))

        assert values == pytest.approx([
            ((967, 143, 1084, 261), 0.851),
            ((246, 208, 331, 287), 0.81),
            ((662, 466, 705, 514), 0.733),
            ((479, 283, 523, 326), 0.72),
        ])

    def test_detect_faces_none(self):
        assert detect_faces(get_testfile('png_full.png')) == []
