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
class TestPreprocessTransformersSiglip:
    @skipUnless(_HAS_TRANSFORMERS, 'Transformers required.')
    @pytest.mark.parametrize(*tmatrix({
        'repo_id': [
            'Marqo/marqo-ecommerce-embeddings-B',
            'ucsahin/TraVisionLM-DPO',
            'llava-hf/llava-interleave-qwen-0.5b-hf',
            'zhumj34/Mipha-3B',
            'lmms-lab/llava-onevision-qwen2-72b-ov-sft',
            'p1atdev/siglip-tagger-test-3',
            'gokaygokay/paligemma-rich-captions',
            'lmms-lab/llava-onevision-qwen2-0.5b-ov',
            'gokaygokay/sd3-long-captioner-v2',
            'OpenFace-CQUPT/Human_LLaVA',
            'ucsahin/TraVisionLM-base',
            'mlx-community/paligemma-3b-mix-448-8bit',

            'google/siglip-so400m-patch14-384',
            'google/siglip-so400m-patch14-224',
            'google/siglip-base-patch16-256-multilingual',
            'google/siglip-base-patch16-224',
            'google/siglip-so400m-patch16-256-i18n',
            'google/siglip-base-patch16-512',
            'google/siglip-large-patch16-256',
            'google/siglip-base-patch16-384',
            'google/siglip-large-patch16-384',
            'google/siglip-base-patch16-256'
        ],
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
            'nude_girl.png',
            'dori_640.png',
            'nian_640.png',
        ]
    }))
    def test_siglip_image_preprocess_align(self, src_image, repo_id):
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
