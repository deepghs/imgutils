from unittest import skipUnless

import numpy as np
import pytest
from hbutils.testing import tmatrix

from imgutils.data import load_image
from imgutils.preprocess.transformers import create_transforms_from_transformers
from test.testings import get_testfile

try:
    import transformers
except (ImportError, ModuleNotFoundError):
    _HAS_TRANSFORMERS = False
else:
    _HAS_TRANSFORMERS = True


@pytest.mark.unittest
class TestPreprocessTransformersClip:
    @skipUnless(_HAS_TRANSFORMERS, 'Transformers required.')
    @pytest.mark.parametrize(*tmatrix({
        'repo_id': [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
            "openai/clip-vit-large-patch14-336",
            "microsoft/Florence-2-large",
            "llava-hf/llava-1.5-7b-hf",
            'facebook/metaclip-b32-400m',
            # 'apple/aimv2-large-patch14-224',
            'nvidia/RADIO',
        ],
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
            'nude_girl.png',
            'dori_640.png',
            'nian_640.png',
        ]
    }))
    def test_clip_image_preprocess_align(self, src_image, repo_id):
        from transformers import AutoImageProcessor
        image = load_image(get_testfile(src_image), mode='RGB', force_background='white')
        processor = AutoImageProcessor.from_pretrained(repo_id)

        trans = create_transforms_from_transformers(processor)

        expected_output = processor.preprocess(image)['pixel_values'][0]
        output = trans(image)
        np.testing.assert_array_almost_equal(
            output,
            expected_output,
        )

    @skipUnless(_HAS_TRANSFORMERS, 'Transformers required.')
    @pytest.mark.parametrize(*tmatrix({
        'repo_id': [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
            "openai/clip-vit-large-patch14-336",
        ],
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
            'nude_girl.png',
            'dori_640.png',
            'nian_640.png',
        ]
    }))
    def test_clip_preprocess_align(self, src_image, repo_id):
        from transformers import AutoProcessor
        image = load_image(get_testfile(src_image), mode='RGB', force_background='white')
        processor = AutoProcessor.from_pretrained(repo_id)

        trans = create_transforms_from_transformers(processor)

        expected_output = processor.image_processor.preprocess(image)['pixel_values'][0]
        output = trans(image)
        np.testing.assert_array_almost_equal(
            output,
            expected_output,
        )
