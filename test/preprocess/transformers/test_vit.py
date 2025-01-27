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
class TestPreprocessTransformersViT:
    @skipUnless(_HAS_TRANSFORMERS, 'Transformers required.')
    @pytest.mark.parametrize(*tmatrix({
        'repo_id': [
            "Falconsai/nsfw_image_detection",
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-base-printed",
            "dima806/facial_emotions_image_detection",
            "rizvandwiki/gender-classification",
            "AdamCodd/vit-base-nsfw-detector",
            "MixTex/ZhEn-Latex-OCR",
            "prithivMLmods/Deep-Fake-Detector-Model",
        ],
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
            'nude_girl.png',
            'dori_640.png',
            'nian_640.png',
        ]
    }))
    def test_convnext_image_preprocess_align(self, src_image, repo_id):
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
