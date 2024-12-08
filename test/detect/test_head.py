import pytest

from imgutils.detect import detection_similarity
from imgutils.detect.head import detect_heads
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
    def test_detect_heads(self):
        detections = detect_heads(get_testfile('genshin_post.jpg'))
        assert len(detections) == 4

        assert detection_similarity(
            detections,
            [
                ((211, 162, 349, 287), 'head', 0.897052526473999),
                ((938, 88, 1124, 260), 'head', 0.8672211170196533),
                ((461, 251, 531, 325), 'head', 0.8515472412109375),
                ((652, 440, 727, 513), 'head', 0.8476731181144714),
            ]
        ) >= 0.9

    def test_detect_heads_none(self):
        assert detect_heads(get_testfile('png_full.png')) == []

    def test_detect_heads_not_found(self):
        with pytest.raises(ValueError):
            _ = detect_heads(get_testfile('genshin_post.png'), model_name='not_found')

    @pytest.mark.parametrize(['model_name'], [
        ('head_detect_v1.6_n_yv10',),
    ])
    def test_detect_with_yolov10(self, model_name: str):
        detections = detect_heads(get_testfile('genshin_post.jpg'), model_name=model_name)
        similarity = detection_similarity(detections, [
            ((211, 161, 350, 288), 'head', 0.9167004823684692),
            ((938, 87, 1122, 261), 'head', 0.9030089378356934),
            ((463, 249, 532, 327), 'head', 0.8779274225234985),
            ((653, 444, 722, 512), 'head', 0.8771967887878418)
        ])
        assert similarity >= 0.85

    @pytest.mark.parametrize(['model_name'], [
        ('head_detect_v1.6_l_rtdetr',),
    ])
    def test_detect_with_rtdetr(self, model_name: str):
        # ATTENTION: results of rtdetr models are really shitty and unstable
        #            so this expected result is 100% bullshit
        #            just make sure the rtdetr models can be properly inferred
        detections = detect_heads(get_testfile('genshin_post.jpg'), model_name=model_name)
        similarity = detection_similarity(detections, [
            ((210, 160, 351, 286), 'head', 0.9095626473426819),
            ((937, 87, 1123, 260), 'head', 0.8920016288757324),
            ((461, 248, 536, 326), 'head', 0.8442408442497253),
            ((651, 439, 728, 514), 'head', 0.841901421546936)
        ])
        assert similarity >= 0.85
