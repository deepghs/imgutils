import pytest

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
        detections = detect_censors(get_testfile('nude_girl.png'))
        assert len(detections) == 3

        values = []
        for bbox, label, score in detections:
            assert label in {'nipple_f', 'penis', 'pussy'}
            values.append((bbox, int(score * 1000) / 1000))

        assert values == pytest.approx([
            ((365, 264, 399, 289), 0.747),
            ((224, 260, 252, 285), 0.683),
            ((206, 523, 240, 608), 0.679),
        ])

    def test_detect_censors_none(self):
        assert detect_censors(get_testfile('png_full.png')) == []
