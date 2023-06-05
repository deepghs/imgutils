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
            ((966, 141, 1084, 261), 0.852),
            ((247, 206, 330, 287), 0.799),
            ((479, 282, 523, 327), 0.689),
            ((662, 465, 705, 513), 0.681),
        ])

    def test_detect_faces_none(self):
        assert detect_faces(get_testfile('png_full.png')) == []
