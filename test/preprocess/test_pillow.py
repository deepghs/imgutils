from unittest import skipUnless

import pytest
from PIL import Image
from hbutils.testing import tmatrix

from imgutils.preprocess.pillow import PillowResize, _get_pillow_resample
from imgutils.preprocess.torchvision import _get_interpolation_mode
from test.testings import get_testfile

try:
    import torchvision
except (ImportError, ModuleNotFoundError):
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


@pytest.mark.unittest
class TestPreprocessPillow:
    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision unavailable.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_full.png',
            'png_full_m90.png',
        ],
        'size': [
            224,
            384,
        ],
        'resample': [
            'bilinear',
            'bicubic',
            'lanczos',
            'box',
            'hamming',
            'nearest',
        ]
    }))
    def test_resize(self, src_image, size, resample, image_diff):
        from torchvision.transforms import Resize
        image = Image.open(get_testfile(src_image))
        presize = PillowResize(
            size=size,
            interpolation=_get_pillow_resample(resample),
        )
        tresize = Resize(
            size=size,
            interpolation=_get_interpolation_mode(resample),
        )
        assert image_diff(presize(image), tresize(image), throw_exception=False) < 1e-3

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision unavailable.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_full.png',
            'png_full_m90.png',
        ],
        'size': [
            (224, 384),
            (384, 224),
            (256,),
        ],
        'resample': [
            'bilinear',
            'bicubic',
            'lanczos',
            'box',
            'hamming',
            'nearest',
        ]
    }))
    def test_resize_pair_sizes(self, src_image, size, resample, image_diff):
        from torchvision.transforms import Resize
        image = Image.open(get_testfile(src_image))
        presize = PillowResize(
            size=size,
            interpolation=_get_pillow_resample(resample),
        )
        tresize = Resize(
            size=size,
            interpolation=_get_interpolation_mode(resample),
        )
        assert image_diff(presize(image), tresize(image), throw_exception=False) < 1e-3
