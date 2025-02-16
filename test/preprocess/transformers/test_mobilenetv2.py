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
class TestPreprocessTransformersMobilenetV2:
    @skipUnless(_HAS_TRANSFORMERS, 'Transformers required.')
    @pytest.mark.parametrize(*tmatrix({
        'repo_id': [
            'ChispiDEV/autotrain-1tqht-w0zz7',
            'elenaThevalley/mobilenet_v2_1.0_224-finetuned-prueba',
            'cannu/autotrain-4hvd9-vjy72',
            'pradanaadn/mobilenet_v2-activity-recognition',
            'Diginsa/Plant-Disease-Detection-Project',
            'jayanthspratap/mobilenet_v2_1.0_224-cxr-view',
            'sngsfydy/MyMobileNet_v2',
            'nikkopg/102623_mobilenet_v2_1.0_224-finetuned-stucktip',
            'nikkopg/102723-mobilenet_v2_1.0_224-finetuned-stucktip',
            'amiune/clasificacion-bananas',
            'KCAZAR/mi-banana-variedades',
            'aslez123/mobilenet_fashion',
            'nikkopg/102523_mobilenet_v2_1.0_224-finetuned-stucktip',
            'ChispiDEV/autotrain-pky99-ias73',
            'sngsfydy/MobileNetV2_with_Trainer_11_10_2023',

            'google/mobilenet_v2_1.0_224',
            'google/deeplabv3_mobilenet_v2_1.0_513',
            'google/mobilenet_v2_0.75_160',
            'google/mobilenet_v2_1.4_224',
            'google/mobilenet_v2_0.35_96',
        ],
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
            'nude_girl.png',
            'dori_640.png',
            'nian_640.png',
        ]
    }))
    def test_mobilenetv2_image_preprocess_align(self, src_image, repo_id):
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
