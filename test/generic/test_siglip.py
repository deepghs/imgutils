import re

import numpy as np
import pytest

from imgutils.generic import siglip_image_encode, siglip_text_encode, siglip_predict
from imgutils.generic.siglip import _open_models_for_repo_id
from test.testings import get_testfile


@pytest.fixture(scope='module')
def siglip_repo_id():
    return 'deepghs/siglip_onnx'


@pytest.fixture(scope='module')
def siglip_model_name():
    return 'google/siglip-base-patch16-256-multilingual'


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run(siglip_repo_id):
    try:
        yield
    finally:
        _open_models_for_repo_id(siglip_repo_id).clear()


@pytest.mark.unittest
class TestGenericSiglip:
    @pytest.mark.parametrize(['name'], [
        ('unsplash_sZzmhn2xjQY',),
        ('unsplash_S-8ntPEsSwo',),
        ('unsplash_tB4-ftQ4zyI',),
        ('unsplash_l6KamCXeB4U',),
        ('unsplash__9dAwWA4LD8',),
        ('unsplash_LlsAieNJE70',),
        ('unsplash_HWIOLU7_O6w',),
        ('unsplash_1AAa78W_Ezc',),
        ('unsplash_0TPmrjTXjSs',),
        ('unsplash_0yAVtZiYkJY',)
    ])
    def test_siglip_image_encode(self, name, siglip_repo_id, siglip_model_name):
        src_image = get_testfile('dataset', 'unsplash_1000', f'{name}.jpg')
        dst_npy = get_testfile('siglip', 'unsplash_1000', f'{name}.npy')
        embedding = siglip_image_encode(src_image, repo_id=siglip_repo_id, model_name=siglip_model_name)
        expected_embedding = np.load(dst_npy)
        np.testing.assert_allclose(embedding, expected_embedding, rtol=1e-03, atol=1e-05)

    @pytest.mark.parametrize(['text'], [
        ("a red car parked on the street",),
        ("beautiful sunset over mountain landscape",),
        ("two cats playing with yarn",),
        ("fresh fruits in a wooden bowl",),
        ("person reading book under tree",),
        ("colorful hot air balloon in blue sky",),
        ("children playing soccer in the park",),
        ("rustic cabin surrounded by pine trees",),
        ("waves crashing on sandy beach",),
        ("chef cooking in modern kitchen",),
    ])
    def test_siglip_text_encode(self, text, siglip_repo_id, siglip_model_name):
        dst_npy = get_testfile('siglip', 'text', re.sub(r'[\W_]+', '_', text).strip('_') + '.npy')
        embedding = siglip_text_encode(text, repo_id=siglip_repo_id, model_name=siglip_model_name)
        expected_embedding = np.load(dst_npy)
        np.testing.assert_allclose(embedding, expected_embedding, rtol=1e-03, atol=1e-05)

    def test_siglip_predict(self, siglip_repo_id, siglip_model_name):
        result = siglip_predict(
            images=[
                get_testfile('clip_cats.jpg'),
                get_testfile('idolsankaku', '3.jpg'),
            ],
            texts=[
                'a photo of a cat',
                'a photo of 2 cats',
                'a photo of 2 dogs',
                'a photo of a woman',
            ],
            repo_id=siglip_repo_id,
            model_name=siglip_model_name,
        )
        expected_result = np.array(
            [[0.0013782851165160537, 0.27010253071784973, 9.751768811838701e-05, 3.6702780814579228e-09],
             [1.2790776438009743e-08, 4.396981001519862e-09, 3.2838454178119036e-10, 1.0559210750216153e-06]])
        np.testing.assert_allclose(result, expected_result, atol=3e-4)
