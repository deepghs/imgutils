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
class TestPreprocessTransformersBlip:
    @skipUnless(_HAS_TRANSFORMERS, 'Transformers required.')
    @pytest.mark.parametrize(*tmatrix({
        'repo_id': [
            'blackhole33/Image2text',
            'StanfordAIMI/XrayCLIP__vit-b-16__laion2b-s34b-b88k',
            'gizmo-ai/blip-image-captioning-large',
            'sooh-j/blip2-vizwizqa',
            'ethzanalytics/blip2-flan-t5-xl-sharded',
            'dblasko/blip-dalle3-img2prompt',
            'dineshcr7/Final-BLIP-LORA',
            'advaitadasein/blip2-opt-6.7b',
            'moranyanuka/blip-image-captioning-base-mocha',
            'upro/blip',
            'Revrse/icon-captioning-model',
            'Yhyu13/instructblip-vicuna-7b-gptq-4bit',
            'moranyanuka/blip-image-captioning-large-mocha',
            'Mediocreatmybest/blip2-opt-2.7b_8bit',
            'ybelkada/blip-image-captioning-base-football-finetuned',

            'Salesforce/blip-image-captioning-large',
            'Salesforce/blip-image-captioning-base',
            'Salesforce/blip2-opt-2.7b',
            'Salesforce/blip-vqa-base',
            'Salesforce/instructblip-vicuna-7b',
            'Salesforce/blip2-flan-t5-xxl',
            'Salesforce/blip2-opt-6.7b',
            'Salesforce/blip2-flan-t5-xl',
            'Salesforce/blip-vqa-capfilt-large',
            'Salesforce/instructblip-vicuna-13b',
            'Salesforce/blip2-opt-6.7b-coco',
            'Salesforce/instructblip-flan-t5-xl',
            'Salesforce/instructblip-flan-t5-xxl',
            'Salesforce/blip-itm-base-coco',
            'Salesforce/blip2-flan-t5-xl-coco',
            'Salesforce/blip2-opt-2.7b-coco',
            'Salesforce/blip-itm-large-flickr',
            'Salesforce/blip2-itm-vit-g-coco',
            'Salesforce/blip2-itm-vit-g',
            'Salesforce/blip-itm-base-flickr',
            'Salesforce/blip-itm-large-coco'
        ],
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
            'nude_girl.png',
            'dori_640.png',
            'nian_640.png',
        ]
    }))
    def test_blip_image_preprocess_align(self, src_image, repo_id):
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
