import pytest
from PIL import Image

from imgutils.detect import detect_with_booru_yolo, detection_similarity
from imgutils.generic.yolo import _open_models_for_repo_id
from ..testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.fixture()
def nude_girl_file():
    return get_testfile('nude_girl.png')


@pytest.fixture()
def nude_girl_image(nude_girl_file):
    return Image.open(nude_girl_file)


@pytest.fixture()
def nude_girl_detection():
    return [
        ((243, 0, 444, 253), 'head', 0.9584344029426575),
        ((213, 231, 426, 358), 'boob', 0.9308794140815735),
        ((86, 393, 401, 701), 'sprd', 0.8639463186264038)
    ]


@pytest.mark.unittest
class TestDetectBooruYOLO:
    def test_detect_with_booru_yolo_file(self, nude_girl_file, nude_girl_detection):
        detection = detect_with_booru_yolo(nude_girl_file)
        similarity = detection_similarity(detection, nude_girl_detection)
        assert similarity >= 0.9

    def test_detect_with_booru_yolo_image(self, nude_girl_image, nude_girl_detection):
        detection = detect_with_booru_yolo(nude_girl_image)
        similarity = detection_similarity(detection, nude_girl_detection)
        assert similarity >= 0.9
