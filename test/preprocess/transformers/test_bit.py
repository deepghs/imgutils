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
class TestPreprocessTransformersBit:
    @skipUnless(_HAS_TRANSFORMERS, 'Transformers required.')
    @pytest.mark.parametrize(*tmatrix({
        'repo_id': [
            'facebook/dinov2-small-imagenet1k-1-layer',
            'robertsw/aesthetics_v2',
            'facebook/hiera-huge-224-mae-hf',
            'Kalinga/dinov2-base-finetuned-oxford',
            'facebook/dinov2-with-registers-giant',
            'facebook/hiera-base-224-in1k-hf',
            'zkatona/dinov2-base-finetuned-oxford',
            'microsoft/focalnet-small',
            'atuo/vit-base-patch16-224-in21k-finetuned-crop-classification',
            'facebook/hiera-tiny-224-in1k-hf',
            'facebook/hiera-tiny-224-mae-hf',
            'microsoft/rad-dino',
            'facebook/dinov2-with-registers-base',
            'NamLe12/vit-base-beans',
            'suncy13/Foot',

            'facebook/dinov2-base',
            'facebook/dinov2-large',
            'facebook/dinov2-giant',
            'facebook/dinov2-small',
            'facebook/dinov2-base-imagenet1k-1-layer',
            'facebook/dinov2-with-registers-giant',
            'facebook/dinov2-with-registers-small',
            'facebook/dinov2-giant-imagenet1k-1-layer',
            'facebook/dinov2-with-registers-large',
            'facebook/dinov2-with-registers-base',
            'facebook/dinov2-large-imagenet1k-1-layer',
            'facebook/dinov2-small-imagenet1k-1-layer',
            'facebook/dinov2-with-registers-giant-imagenet1k-1-layer',
            'facebook/dinov2-with-registers-base-imagenet1k-1-layer',
            'facebook/dinov2-with-registers-large-imagenet1k-1-layer',
            'facebook/dinov2-with-registers-small-imagenet1k-1-layer',
        ],
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
            'nude_girl.png',
            'dori_640.png',
            'nian_640.png',
        ]
    }))
    def test_bit_image_preprocess_align(self, src_image, repo_id):
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
