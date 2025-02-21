import re

import numpy as np
import pytest

from imgutils.generic.clip import _open_models_for_repo_id, clip_image_encode, clip_text_encode, clip_predict
from test.testings import get_testfile


@pytest.fixture(scope='module')
def clip_repo_id():
    return 'deepghs/clip_onnx'


@pytest.fixture(scope='module')
def clip_model_name():
    return 'openai/clip-vit-base-patch32'


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run(clip_repo_id):
    try:
        yield
    finally:
        _open_models_for_repo_id(clip_repo_id).clear()


@pytest.mark.unittest
class TestGenericCLIP:
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
    def test_clip_image_encode(self, name, clip_repo_id, clip_model_name):
        src_image = get_testfile('dataset', 'unsplash_1000', f'{name}.jpg')
        dst_npy = get_testfile('clip', 'unsplash_1000', f'{name}.npy')
        embedding = clip_image_encode(src_image, repo_id=clip_repo_id, model_name=clip_model_name)
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
    def test_clip_text_encode(self, text, clip_repo_id, clip_model_name):
        dst_npy = get_testfile('clip', 'text', re.sub(r'[\W_]+', '_', text).strip('_') + '.npy')
        embedding = clip_text_encode(text, repo_id=clip_repo_id, model_name=clip_model_name)
        expected_embedding = np.load(dst_npy)
        np.testing.assert_allclose(embedding, expected_embedding, rtol=1e-03, atol=1e-05)

    def test_clip_predict(self, clip_repo_id, clip_model_name):
        result = clip_predict(
            images=[
                get_testfile('clip_cats.jpg'),
                get_testfile('idolsankaku', '3.jpg'),
            ],
            texts=[
                'a photo of a cat',
                'a photo of a dog',
                'a photo of a human',
            ],
            repo_id=clip_repo_id,
            model_name=clip_model_name,
        )
        expected_result = np.array([[0.9803991317749023, 0.005067288409918547, 0.01453354675322771],
                                    [0.21404513716697693, 0.049479320645332336, 0.7364755272865295]])
        np.testing.assert_allclose(result, expected_result, atol=3e-4)
