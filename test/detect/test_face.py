import pytest

from imgutils.detect.face import _open_face_detect_model, detect_faces
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_face_detect_model.cache_clear()


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
            ((963, 139, 1087, 265), 0.847),
            ((242, 205, 336, 290), 0.807),
            ((477, 279, 526, 330), 0.7),
            ((658, 464, 708, 513), 0.699),
        ])

    def test_detect_faces_none(self):
        assert detect_faces(get_testfile('png_full.png')) == []
