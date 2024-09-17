import pytest
from PIL import Image

from imgutils.detect import detect_with_booru_yolo
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
        ((236, 1, 452, 247), 'head', 0.9584360718727112),
        ((211, 236, 431, 346), 'boob', 0.9300149083137512),
        ((62, 402, 427, 697), 'sprd', 0.8708215951919556)
    ]


@pytest.mark.unittest
class TestDetectBooruYOLO:
    def test_detect_with_booru_yolo_file(self, nude_girl_file, nude_girl_detection):
        detection = detect_with_booru_yolo(nude_girl_file)
        assert [label for _, label, _ in detection] == \
               [label for _, label, _ in nude_girl_detection]
        for (actual_box, _, _), (expected_box, _, _) in zip(detection, nude_girl_detection):
            assert actual_box == pytest.approx(expected_box)
        assert [score for _, _, score in detection] == \
               pytest.approx([score for _, _, score in nude_girl_detection], abs=1e-4)

    def test_detect_with_booru_yolo_image(self, nude_girl_image, nude_girl_detection):
        detection = detect_with_booru_yolo(nude_girl_image)
        assert [label for _, label, _ in detection] == \
               [label for _, label, _ in nude_girl_detection]
        for (actual_box, _, _), (expected_box, _, _) in zip(detection, nude_girl_detection):
            assert actual_box == pytest.approx(expected_box)
        assert [score for _, _, score in detection] == \
               pytest.approx([score for _, _, score in nude_girl_detection], abs=1e-4)
