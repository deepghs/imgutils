from unittest.mock import patch

import numpy as np
import pytest
from huggingface_hub import configure_http_backend
from huggingface_hub.utils import reset_sessions

from imgutils.detect import detection_similarity, detection_with_mask_similarity
from imgutils.generic.yoloseg import _open_models_for_repo_id, yolo_seg_predict
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.fixture()
def clean_session():
    reset_sessions()
    _open_models_for_repo_id.cache_clear()
    try:
        yield
    finally:
        reset_sessions()
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestGenericYOLOSeg:
    def test_detect_text_blocks_with_masks_1(self):
        detection = yolo_seg_predict(
            get_testfile('ml1.png'),
            repo_id='deepghs/segs',
            model_name='vp2c0.3_735k_bs512_seed0_s_yv11',
        )
        bbox_similarity = detection_similarity(detection, [
            ((865, 43, 959, 74), 'text_block', 0.5545626878738403),
            ((693, 100, 774, 130), 'text_block', 0.5359622240066528),
            ((221, 63, 315, 109), 'text_block', 0.4336417019367218),
            ((418, 74, 514, 118), 'text_block', 0.4025779068470001),
        ])
        assert bbox_similarity >= 0.9

        expected_masked = [
            ((0, 0, 0, 0), 'text_block', 0.0, np.load(get_testfile(f'ml1_mask{i}.npy')))
            for i in range(4)
        ]
        mask_similarity = detection_with_mask_similarity(detection, expected_masked)
        assert mask_similarity >= 0.9

    def test_detect_text_blocks_with_masks_none(self):
        assert yolo_seg_predict(
            get_testfile('png_full.png'),
            repo_id='deepghs/segs',
            model_name='vp2c0.3_735k_bs512_seed0_s_yv11',
        ) == []

    @patch("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    def test_detect_text_blocks_with_masks_1_on_offline_mode(self):
        configure_http_backend()
        detection = yolo_seg_predict(
            get_testfile('ml1.png'),
            repo_id='deepghs/segs',
            model_name='vp2c0.3_735k_bs512_seed0_s_yv11',
        )
        bbox_similarity = detection_similarity(detection, [
            ((865, 43, 959, 74), 'text_block', 0.5545626878738403),
            ((693, 100, 774, 130), 'text_block', 0.5359622240066528),
            ((221, 63, 315, 109), 'text_block', 0.4336417019367218),
            ((418, 74, 514, 118), 'text_block', 0.4025779068470001),
        ])
        assert bbox_similarity >= 0.9

        expected_masked = [
            ((0, 0, 0, 0), 'text_block', 0.0, np.load(get_testfile(f'ml1_mask{i}.npy')))
            for i in range(4)
        ]
        mask_similarity = detection_with_mask_similarity(detection, expected_masked)
        assert mask_similarity >= 0.9
