import pytest
from PIL import Image

from imgutils.detect import detect_with_nudenet
from imgutils.detect.nudenet import _open_nudenet_nms, _open_nudenet_yolo
from ..testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_nudenet_yolo.cache_clear()
        _open_nudenet_nms.cache_clear()


@pytest.fixture()
def nude_girl_file():
    return get_testfile('nude_girl.png')


@pytest.fixture()
def nude_girl_image(nude_girl_file):
    return Image.open(nude_girl_file)


@pytest.fixture()
def nude_girl_detection():
    return [
        ((321.3878631591797, 242.3542022705078, 429.8410186767578, 345.7248992919922),
         'FEMALE_BREAST_EXPOSED',
         0.832775890827179),
        ((207.8404312133789, 243.68451690673828, 307.2947006225586, 336.3175582885742),
         'FEMALE_BREAST_EXPOSED',
         0.8057667016983032),
        ((203.23711395263672,
          348.42012786865234,
          351.32117462158203,
          511.34781646728516),
         'BELLY_EXPOSED',
         0.7703637480735779),
        ((280.81117248535156,
          678.6565170288086,
          436.11827087402344,
          767.8816909790039),
         'FEET_EXPOSED',
         0.747696578502655),
        ((185.25140380859375, 518.0437889099121, 252.96240234375, 625.8465919494629),
         'FEMALE_GENITALIA_EXPOSED',
         0.7381105422973633),
        ((287.9706840515137, 124.07051467895508, 392.7693061828613, 225.3848991394043),
         'FACE_FEMALE',
         0.6556487083435059),
        ((103.20288848876953,
          564.7838439941406,
          352.05843353271484,
          707.6390075683594),
         'BUTTOCKS_EXPOSED',
         0.44306617975234985),
        ((396.1982898712158, 224.24786376953125, 450.53956413269043, 290.279541015625),
         'ARMPITS_EXPOSED',
         0.31386712193489075)
    ]


@pytest.mark.unittest
class TestDetectNudeNet:
    def test_detect_with_nudenet_file(self, nude_girl_file, nude_girl_detection):
        detection = detect_with_nudenet(nude_girl_file)
        assert [label for _, label, _ in detection] == \
               [label for _, label, _ in nude_girl_detection]
        for (actual_box, _, _), (expected_box, _, _) in zip(detection, nude_girl_detection):
            assert actual_box == pytest.approx(expected_box)
        assert [score for _, _, score in detection] == \
               pytest.approx([score for _, _, score in nude_girl_detection], abs=1e-4)

    def test_detect_with_nudenet_image(self, nude_girl_image, nude_girl_detection):
        detection = detect_with_nudenet(nude_girl_image)
        assert [label for _, label, _ in detection] == \
               [label for _, label, _ in nude_girl_detection]
        for (actual_box, _, _), (expected_box, _, _) in zip(detection, nude_girl_detection):
            assert actual_box == pytest.approx(expected_box)
        assert [score for _, _, score in detection] == \
               pytest.approx([score for _, _, score in nude_girl_detection], abs=1e-4)
