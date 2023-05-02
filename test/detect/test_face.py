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
class TestDetectFace:
    def test_detect_faces(self):
        detections = detect_faces(get_testfile('genshin_post.jpg'))
        assert len(detections) == 4

        values = []
        for bbox, label, score in detections:
            assert label == 'head'
            values.append((bbox, int(score * 1000) / 1000))

        assert values == pytest.approx([
            ((202, 155, 356, 294), 0.878),
            ((938, 87, 1121, 262), 0.846),
            ((652, 440, 725, 514), 0.839),
            ((464, 250, 535, 326), 0.765)
        ])

    def test_detect_faces_none(self):
        assert detect_faces(get_testfile('png_full.png')) == []
