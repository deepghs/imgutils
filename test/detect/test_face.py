import pytest

from imgutils.detect import detection_similarity
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
        detection = detect_faces(get_testfile('genshin_post.jpg'))
        similarity = detection_similarity(detection, [
            ((966, 142, 1085, 261), 'face', 0.850458025932312),
            ((247, 209, 330, 288), 'face', 0.8288277387619019),
            ((661, 467, 706, 512), 'face', 0.754958987236023),
            ((481, 282, 522, 325), 'face', 0.7148504257202148)
        ])
        assert similarity >= 0.9

    def test_detect_faces_none(self):
        assert detect_faces(get_testfile('png_full.png')) == []
