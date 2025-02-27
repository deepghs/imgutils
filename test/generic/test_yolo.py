from unittest.mock import patch

import pytest
from huggingface_hub import configure_http_backend
from huggingface_hub.utils import reset_sessions

from imgutils.detect import detection_similarity
from imgutils.generic.yolo import _open_models_for_repo_id, yolo_predict
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.fixture(scope='function', autouse=True)
def _clean_session():
    reset_sessions()
    _open_models_for_repo_id.cache_clear()
    print('clean session')
    try:
        yield
    finally:
        reset_sessions()
        _open_models_for_repo_id.cache_clear()
        print('clean session')


@pytest.mark.unittest
class TestGenericYOLO:
    def test_detect_faces(self):
        detection = yolo_predict(
            get_testfile('genshin_post.jpg'),
            repo_id='deepghs/anime_face_detection',
            model_name='face_detect_v1.4_s',
        )
        similarity = detection_similarity(detection, [
            ((966, 142, 1085, 261), 'face', 0.850458025932312),
            ((247, 209, 330, 288), 'face', 0.8288277387619019),
            ((661, 467, 706, 512), 'face', 0.754958987236023),
            ((481, 282, 522, 325), 'face', 0.7148504257202148)
        ])
        assert similarity >= 0.9

    def test_detect_faces_none(self):
        assert yolo_predict(
            get_testfile('png_full.png'),
            repo_id='deepghs/anime_face_detection',
            model_name='face_detect_v1.4_s',
        ) == []

    @patch("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    def test_detect_faces_with_offline_mode(self):
        configure_http_backend()
        detection = yolo_predict(
            get_testfile('genshin_post.jpg'),
            repo_id='deepghs/anime_face_detection',
            model_name='face_detect_v1.4_s',
        )
        similarity = detection_similarity(detection, [
            ((966, 142, 1085, 261), 'face', 0.850458025932312),
            ((247, 209, 330, 288), 'face', 0.8288277387619019),
            ((661, 467, 706, 512), 'face', 0.754958987236023),
            ((481, 282, 522, 325), 'face', 0.7148504257202148)
        ])
        assert similarity >= 0.9
