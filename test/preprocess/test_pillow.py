import os
from unittest import skipUnless

import numpy as np
import pytest
from PIL import Image
from hbutils.testing import tmatrix

from imgutils.data import load_image, grid_transparent
from imgutils.preprocess.pillow import PillowResize, _get_pillow_resample, PillowCenterCrop, PillowToTensor, \
    PillowMaybeToTensor, PillowNormalize, create_pillow_transforms, parse_pillow_transforms, PillowCompose, \
    PillowConvertRGB, PillowRescale, PillowPadToSize
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
    def test_get_pillow_resample_int(self):
        assert _get_pillow_resample(0) == Image.NEAREST
        assert _get_pillow_resample(1) == Image.LANCZOS
        assert _get_pillow_resample(2) == Image.BILINEAR
        assert _get_pillow_resample(3) == Image.BICUBIC
        assert _get_pillow_resample(4) == Image.BOX
        assert _get_pillow_resample(5) == Image.HAMMING

    def test_get_pillow_resample_str(self):
        assert _get_pillow_resample('nearest') == Image.NEAREST
        assert _get_pillow_resample('NEAREST') == Image.NEAREST
        assert _get_pillow_resample('bilinear') == Image.BILINEAR
        assert _get_pillow_resample('bicubic') == Image.BICUBIC
        assert _get_pillow_resample('box') == Image.BOX
        assert _get_pillow_resample('hamming') == Image.HAMMING
        assert _get_pillow_resample('lanczos') == Image.LANCZOS

    def test_invalid_int(self):
        with pytest.raises(ValueError, match='Invalid interpolation value - 6.'):
            _get_pillow_resample(6)

    def test_invalid_str(self):
        with pytest.raises(ValueError, match='Invalid interpolation value - \'invalid\'.'):
            _get_pillow_resample('invalid')

    def test_invalid_type(self):
        with pytest.raises(TypeError, match='Input type must be int or str, got <class \'float\'>'):
            _get_pillow_resample(1.0)

    def test_resize_invalid(self):
        with pytest.raises(TypeError):
            _ = PillowResize(size='888')
        with pytest.raises(ValueError):
            _ = PillowResize(size=())
        with pytest.raises(ValueError):
            _ = PillowResize(size=(1, 1, 4, 5, 1, 4))
        with pytest.raises(ValueError):
            _ = PillowResize(size=(224, 384), max_size=512)

    def test_resize_invalid_input(self):
        resize = PillowResize(size=640)
        with pytest.raises(TypeError):
            _ = resize(np.random.randn(3, 284, 284))

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision unavailable.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'size': [
            224,
            384,
            640,
            888,
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
            'png_640.png',
            'png_640_m90.png',
        ],
        'size': [
            (224, 384),
            (384, 224),
            (256,),
            (999, 888),
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

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision unavailable.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        ('size', 'max_size'): [
            (224, 384),
            (224, 256),
            (224, 225),
            (256, 384),
            (256, 257),
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
    def test_resize_pair_max_sizes(self, src_image, size, max_size, resample, image_diff):
        from torchvision.transforms import Resize
        image = Image.open(get_testfile(src_image))
        presize = PillowResize(
            size=size,
            interpolation=_get_pillow_resample(resample),
            max_size=max_size,
        )
        tresize = Resize(
            size=size,
            interpolation=_get_interpolation_mode(resample),
            max_size=max_size,
        )
        assert image_diff(presize(image), tresize(image), throw_exception=False) < 1e-3

    @pytest.mark.parametrize(['size', 'interpolation', 'max_size', 'antialias', 'repr_text'], [
        (224, 0, None, True, 'PillowResize(size=224, interpolation=nearest, max_size=None, antialias=True)'),
        (224, 0, None, False, 'PillowResize(size=224, interpolation=nearest, max_size=None, antialias=False)'),
        (224, 2, None, True, 'PillowResize(size=224, interpolation=bilinear, max_size=None, antialias=True)'),
        (224, 2, None, False, 'PillowResize(size=224, interpolation=bilinear, max_size=None, antialias=False)'),
        (224, 3, None, True, 'PillowResize(size=224, interpolation=bicubic, max_size=None, antialias=True)'),
        (224, 3, None, False, 'PillowResize(size=224, interpolation=bicubic, max_size=None, antialias=False)'),
        (224, 4, None, True, 'PillowResize(size=224, interpolation=box, max_size=None, antialias=True)'),
        (224, 4, None, False, 'PillowResize(size=224, interpolation=box, max_size=None, antialias=False)'),
        (224, 5, None, True, 'PillowResize(size=224, interpolation=hamming, max_size=None, antialias=True)'),
        (224, 5, None, False, 'PillowResize(size=224, interpolation=hamming, max_size=None, antialias=False)'),
        (224, 1, None, True, 'PillowResize(size=224, interpolation=lanczos, max_size=None, antialias=True)'),
        (224, 1, None, False, 'PillowResize(size=224, interpolation=lanczos, max_size=None, antialias=False)'),
        (224, 0, 384, True, 'PillowResize(size=224, interpolation=nearest, max_size=384, antialias=True)'),
        (224, 0, 384, False, 'PillowResize(size=224, interpolation=nearest, max_size=384, antialias=False)'),
        (224, 2, 384, True, 'PillowResize(size=224, interpolation=bilinear, max_size=384, antialias=True)'),
        (224, 2, 384, False, 'PillowResize(size=224, interpolation=bilinear, max_size=384, antialias=False)'),
        (224, 3, 384, True, 'PillowResize(size=224, interpolation=bicubic, max_size=384, antialias=True)'),
        (224, 3, 384, False, 'PillowResize(size=224, interpolation=bicubic, max_size=384, antialias=False)'),
        (224, 4, 384, True, 'PillowResize(size=224, interpolation=box, max_size=384, antialias=True)'),
        (224, 4, 384, False, 'PillowResize(size=224, interpolation=box, max_size=384, antialias=False)'),
        (224, 5, 384, True, 'PillowResize(size=224, interpolation=hamming, max_size=384, antialias=True)'),
        (224, 5, 384, False, 'PillowResize(size=224, interpolation=hamming, max_size=384, antialias=False)'),
        (224, 1, 384, True, 'PillowResize(size=224, interpolation=lanczos, max_size=384, antialias=True)'),
        (224, 1, 384, False, 'PillowResize(size=224, interpolation=lanczos, max_size=384, antialias=False)'),
        (224, 0, 640, True, 'PillowResize(size=224, interpolation=nearest, max_size=640, antialias=True)'),
        (224, 0, 640, False, 'PillowResize(size=224, interpolation=nearest, max_size=640, antialias=False)'),
        (224, 2, 640, True, 'PillowResize(size=224, interpolation=bilinear, max_size=640, antialias=True)'),
        (224, 2, 640, False, 'PillowResize(size=224, interpolation=bilinear, max_size=640, antialias=False)'),
        (224, 3, 640, True, 'PillowResize(size=224, interpolation=bicubic, max_size=640, antialias=True)'),
        (224, 3, 640, False, 'PillowResize(size=224, interpolation=bicubic, max_size=640, antialias=False)'),
        (224, 4, 640, True, 'PillowResize(size=224, interpolation=box, max_size=640, antialias=True)'),
        (224, 4, 640, False, 'PillowResize(size=224, interpolation=box, max_size=640, antialias=False)'),
        (224, 5, 640, True, 'PillowResize(size=224, interpolation=hamming, max_size=640, antialias=True)'),
        (224, 5, 640, False, 'PillowResize(size=224, interpolation=hamming, max_size=640, antialias=False)'),
        (224, 1, 640, True, 'PillowResize(size=224, interpolation=lanczos, max_size=640, antialias=True)'),
        (224, 1, 640, False, 'PillowResize(size=224, interpolation=lanczos, max_size=640, antialias=False)'),
        (224, 0, 888, True, 'PillowResize(size=224, interpolation=nearest, max_size=888, antialias=True)'),
        (224, 0, 888, False, 'PillowResize(size=224, interpolation=nearest, max_size=888, antialias=False)'),
        (224, 2, 888, True, 'PillowResize(size=224, interpolation=bilinear, max_size=888, antialias=True)'),
        (224, 2, 888, False, 'PillowResize(size=224, interpolation=bilinear, max_size=888, antialias=False)'),
        (224, 3, 888, True, 'PillowResize(size=224, interpolation=bicubic, max_size=888, antialias=True)'),
        (224, 3, 888, False, 'PillowResize(size=224, interpolation=bicubic, max_size=888, antialias=False)'),
        (224, 4, 888, True, 'PillowResize(size=224, interpolation=box, max_size=888, antialias=True)'),
        (224, 4, 888, False, 'PillowResize(size=224, interpolation=box, max_size=888, antialias=False)'),
        (224, 5, 888, True, 'PillowResize(size=224, interpolation=hamming, max_size=888, antialias=True)'),
        (224, 5, 888, False, 'PillowResize(size=224, interpolation=hamming, max_size=888, antialias=False)'),
        (224, 1, 888, True, 'PillowResize(size=224, interpolation=lanczos, max_size=888, antialias=True)'),
        (224, 1, 888, False, 'PillowResize(size=224, interpolation=lanczos, max_size=888, antialias=False)'),
        (384, 0, None, True, 'PillowResize(size=384, interpolation=nearest, max_size=None, antialias=True)'),
        (384, 0, None, False, 'PillowResize(size=384, interpolation=nearest, max_size=None, antialias=False)'),
        (384, 2, None, True, 'PillowResize(size=384, interpolation=bilinear, max_size=None, antialias=True)'),
        (384, 2, None, False, 'PillowResize(size=384, interpolation=bilinear, max_size=None, antialias=False)'),
        (384, 3, None, True, 'PillowResize(size=384, interpolation=bicubic, max_size=None, antialias=True)'),
        (384, 3, None, False, 'PillowResize(size=384, interpolation=bicubic, max_size=None, antialias=False)'),
        (384, 4, None, True, 'PillowResize(size=384, interpolation=box, max_size=None, antialias=True)'),
        (384, 4, None, False, 'PillowResize(size=384, interpolation=box, max_size=None, antialias=False)'),
        (384, 5, None, True, 'PillowResize(size=384, interpolation=hamming, max_size=None, antialias=True)'),
        (384, 5, None, False, 'PillowResize(size=384, interpolation=hamming, max_size=None, antialias=False)'),
        (384, 1, None, True, 'PillowResize(size=384, interpolation=lanczos, max_size=None, antialias=True)'),
        (384, 1, None, False, 'PillowResize(size=384, interpolation=lanczos, max_size=None, antialias=False)'),
        (384, 0, 384, True, 'PillowResize(size=384, interpolation=nearest, max_size=384, antialias=True)'),
        (384, 0, 384, False, 'PillowResize(size=384, interpolation=nearest, max_size=384, antialias=False)'),
        (384, 2, 384, True, 'PillowResize(size=384, interpolation=bilinear, max_size=384, antialias=True)'),
        (384, 2, 384, False, 'PillowResize(size=384, interpolation=bilinear, max_size=384, antialias=False)'),
        (384, 3, 384, True, 'PillowResize(size=384, interpolation=bicubic, max_size=384, antialias=True)'),
        (384, 3, 384, False, 'PillowResize(size=384, interpolation=bicubic, max_size=384, antialias=False)'),
        (384, 4, 384, True, 'PillowResize(size=384, interpolation=box, max_size=384, antialias=True)'),
        (384, 4, 384, False, 'PillowResize(size=384, interpolation=box, max_size=384, antialias=False)'),
        (384, 5, 384, True, 'PillowResize(size=384, interpolation=hamming, max_size=384, antialias=True)'),
        (384, 5, 384, False, 'PillowResize(size=384, interpolation=hamming, max_size=384, antialias=False)'),
        (384, 1, 384, True, 'PillowResize(size=384, interpolation=lanczos, max_size=384, antialias=True)'),
        (384, 1, 384, False, 'PillowResize(size=384, interpolation=lanczos, max_size=384, antialias=False)'),
        (384, 0, 640, True, 'PillowResize(size=384, interpolation=nearest, max_size=640, antialias=True)'),
        (384, 0, 640, False, 'PillowResize(size=384, interpolation=nearest, max_size=640, antialias=False)'),
        (384, 2, 640, True, 'PillowResize(size=384, interpolation=bilinear, max_size=640, antialias=True)'),
        (384, 2, 640, False, 'PillowResize(size=384, interpolation=bilinear, max_size=640, antialias=False)'),
        (384, 3, 640, True, 'PillowResize(size=384, interpolation=bicubic, max_size=640, antialias=True)'),
        (384, 3, 640, False, 'PillowResize(size=384, interpolation=bicubic, max_size=640, antialias=False)'),
        (384, 4, 640, True, 'PillowResize(size=384, interpolation=box, max_size=640, antialias=True)'),
        (384, 4, 640, False, 'PillowResize(size=384, interpolation=box, max_size=640, antialias=False)'),
        (384, 5, 640, True, 'PillowResize(size=384, interpolation=hamming, max_size=640, antialias=True)'),
        (384, 5, 640, False, 'PillowResize(size=384, interpolation=hamming, max_size=640, antialias=False)'),
        (384, 1, 640, True, 'PillowResize(size=384, interpolation=lanczos, max_size=640, antialias=True)'),
        (384, 1, 640, False, 'PillowResize(size=384, interpolation=lanczos, max_size=640, antialias=False)'),
        (384, 0, 888, True, 'PillowResize(size=384, interpolation=nearest, max_size=888, antialias=True)'),
        (384, 0, 888, False, 'PillowResize(size=384, interpolation=nearest, max_size=888, antialias=False)'),
        (384, 2, 888, True, 'PillowResize(size=384, interpolation=bilinear, max_size=888, antialias=True)'),
        (384, 2, 888, False, 'PillowResize(size=384, interpolation=bilinear, max_size=888, antialias=False)'),
        (384, 3, 888, True, 'PillowResize(size=384, interpolation=bicubic, max_size=888, antialias=True)'),
        (384, 3, 888, False, 'PillowResize(size=384, interpolation=bicubic, max_size=888, antialias=False)'),
        (384, 4, 888, True, 'PillowResize(size=384, interpolation=box, max_size=888, antialias=True)'),
        (384, 4, 888, False, 'PillowResize(size=384, interpolation=box, max_size=888, antialias=False)'),
        (384, 5, 888, True, 'PillowResize(size=384, interpolation=hamming, max_size=888, antialias=True)'),
        (384, 5, 888, False, 'PillowResize(size=384, interpolation=hamming, max_size=888, antialias=False)'),
        (384, 1, 888, True, 'PillowResize(size=384, interpolation=lanczos, max_size=888, antialias=True)'),
        (384, 1, 888, False, 'PillowResize(size=384, interpolation=lanczos, max_size=888, antialias=False)'),
        ((224, 384), 0, None, True,
         'PillowResize(size=(224, 384), interpolation=nearest, max_size=None, antialias=True)'),
        ((224, 384), 0, None, False,
         'PillowResize(size=(224, 384), interpolation=nearest, max_size=None, antialias=False)'),
        ((224, 384), 2, None, True,
         'PillowResize(size=(224, 384), interpolation=bilinear, max_size=None, antialias=True)'),
        ((224, 384), 2, None, False,
         'PillowResize(size=(224, 384), interpolation=bilinear, max_size=None, antialias=False)'),
        ((224, 384), 3, None, True,
         'PillowResize(size=(224, 384), interpolation=bicubic, max_size=None, antialias=True)'),
        ((224, 384), 3, None, False,
         'PillowResize(size=(224, 384), interpolation=bicubic, max_size=None, antialias=False)'),
        ((224, 384), 4, None, True, 'PillowResize(size=(224, 384), interpolation=box, max_size=None, antialias=True)'),
        (
                (224, 384), 4, None, False,
                'PillowResize(size=(224, 384), interpolation=box, max_size=None, antialias=False)'),
        ((224, 384), 5, None, True,
         'PillowResize(size=(224, 384), interpolation=hamming, max_size=None, antialias=True)'),
        ((224, 384), 5, None, False,
         'PillowResize(size=(224, 384), interpolation=hamming, max_size=None, antialias=False)'),
        ((224, 384), 1, None, True,
         'PillowResize(size=(224, 384), interpolation=lanczos, max_size=None, antialias=True)'),
        ((224, 384), 1, None, False,
         'PillowResize(size=(224, 384), interpolation=lanczos, max_size=None, antialias=False)'),
        ([224, 284], 0, None, True,
         'PillowResize(size=[224, 284], interpolation=nearest, max_size=None, antialias=True)'),
        ([224, 284], 0, None, False,
         'PillowResize(size=[224, 284], interpolation=nearest, max_size=None, antialias=False)'),
        ([224, 284], 2, None, True,
         'PillowResize(size=[224, 284], interpolation=bilinear, max_size=None, antialias=True)'),
        ([224, 284], 2, None, False,
         'PillowResize(size=[224, 284], interpolation=bilinear, max_size=None, antialias=False)'),
        ([224, 284], 3, None, True,
         'PillowResize(size=[224, 284], interpolation=bicubic, max_size=None, antialias=True)'),
        ([224, 284], 3, None, False,
         'PillowResize(size=[224, 284], interpolation=bicubic, max_size=None, antialias=False)'),
        ([224, 284], 4, None, True, 'PillowResize(size=[224, 284], interpolation=box, max_size=None, antialias=True)'),
        (
                [224, 284], 4, None, False,
                'PillowResize(size=[224, 284], interpolation=box, max_size=None, antialias=False)'),
        ([224, 284], 5, None, True,
         'PillowResize(size=[224, 284], interpolation=hamming, max_size=None, antialias=True)'),
        ([224, 284], 5, None, False,
         'PillowResize(size=[224, 284], interpolation=hamming, max_size=None, antialias=False)'),
        ([224, 284], 1, None, True,
         'PillowResize(size=[224, 284], interpolation=lanczos, max_size=None, antialias=True)'),
        ([224, 284], 1, None, False,
         'PillowResize(size=[224, 284], interpolation=lanczos, max_size=None, antialias=False)'),

    ])
    def test_resize_repr(self, size, interpolation, max_size, antialias, repr_text):
        size = PillowResize(
            size=size,
            interpolation=interpolation,
            max_size=max_size,
            antialias=antialias,
        )
        assert repr(size) == repr_text

    def test_center_crop_invalid(self):
        with pytest.raises(ValueError):
            _ = PillowCenterCrop(size='str')

    def test_center_crop_invali_call(self):
        center_crop = PillowCenterCrop(224)
        with pytest.raises(TypeError):
            _ = center_crop(np.random.randn(3, 284, 384))

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision available required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'size': [
            224,
            384,
            888,
        ]
    }))
    def test_center_crop(self, src_image, size, image_diff):
        from torchvision.transforms import CenterCrop
        image = Image.open(get_testfile(src_image))
        pcentercrop = PillowCenterCrop(size=size)
        tcentercrop = CenterCrop(size=size)
        assert image_diff(pcentercrop(image), tcentercrop(image), throw_exception=False) < 1e-3

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision available required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'size': [
            (224,),
            (384,),
            (888,),
            (224, 384),
            (384, 224),
            (384, 888),
            (888, 384),
        ]
    }))
    def test_center_crop_pair(self, src_image, size, image_diff):
        from torchvision.transforms import CenterCrop
        image = Image.open(get_testfile(src_image))
        pcentercrop = PillowCenterCrop(size=size)
        tcentercrop = CenterCrop(size=size)
        assert image_diff(pcentercrop(image), tcentercrop(image), throw_exception=False) < 1e-3

    @pytest.mark.parametrize(['size', 'repr_text'], [
        (224, 'PillowCenterCrop(size=(224, 224))'),
        (384, 'PillowCenterCrop(size=(384, 384))'),
        ((224,), 'PillowCenterCrop(size=(224, 224))'),
        ([384], 'PillowCenterCrop(size=(384, 384))'),
        ((224, 384), 'PillowCenterCrop(size=(224, 384))'),
        ([224, 284], 'PillowCenterCrop(size=(224, 284))'),
    ])
    def test_center_crop_repr(self, size, repr_text):
        assert repr(PillowCenterCrop(size=size)) == repr_text

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'mode': [
            'I', 'I;16', 'F',
            '1', 'L', 'LA', 'P',
            'RGB', 'YCbCr', 'RGBA', 'CMYK',
        ]
    }))
    def test_to_tensor(self, src_image, mode):
        from torchvision.transforms import ToTensor
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode
        ptotensor = PillowToTensor()
        ttotensor = ToTensor()
        np.testing.assert_array_almost_equal(ptotensor(image), ttotensor(image).numpy())

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'mode': [
            'I', 'I;16', 'F',
            '1', 'L', 'LA', 'P',
            'RGB', 'YCbCr', 'RGBA', 'CMYK',
        ]
    }))
    def test_to_tensor(self, src_image, mode):
        from torchvision.transforms import ToTensor
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode
        ptotensor = create_pillow_transforms({'type': 'to_tensor'})
        ttotensor = ToTensor()
        np.testing.assert_array_almost_equal(ptotensor(image), ttotensor(image).numpy())

    def test_to_tensor_invalid(self):
        ptotensor = PillowToTensor()
        with pytest.raises(TypeError):
            _ = ptotensor(np.random.randn(3, 384, 384))

    def test_to_tensor_repr(self):
        assert repr(PillowToTensor()) == 'PillowToTensor()'

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'mode': [
            'I', 'I;16', 'F',
            '1', 'L', 'LA', 'P',
            'RGB', 'YCbCr', 'RGBA', 'CMYK',
        ]
    }))
    def test_maybe_to_tensor(self, src_image, mode):
        from torchvision.transforms import ToTensor
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode
        pmaybetotensor = PillowMaybeToTensor()
        ttotensor = ToTensor()
        np.testing.assert_array_almost_equal(pmaybetotensor(image), ttotensor(image).numpy())

    @pytest.mark.parametrize(['seed'], [
        (i,) for i in range(10)
    ])
    def test_maybe_to_tensor_numpy(self, seed):
        np.random.seed(seed)
        arr = np.random.randn(3, 384, 384)
        pmaybetotensor = PillowMaybeToTensor()
        np.testing.assert_array_almost_equal(arr, pmaybetotensor(arr))

    def test_maybe_to_tensor_repr(self):
        assert repr(PillowMaybeToTensor()) == 'PillowMaybeToTensor()'

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'mode': [
            # 'I', 'I;16', 'F', '1', 'L', 'P',  # 1dim
            # 'LA',  # 2dim
            'RGB', 'YCbCr',  # 3dim
            # 'RGBA', 'CMYK',  # 4dim
        ],
        'mean': [
            (0.4850, 0.4560, 0.4060),
            (0.5, 0.5, 0.5),
            (0.0, 0.0, 0.0),
            (0.4850,),
            (0.5,),
            (0.0,),
        ],
        'std': [
            (0.2290, 0.2240, 0.2250),
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            (0.2290,),
            (0.5,),
            (1.0,),
        ]
    }))
    def test_maybe_normalize_3dim(self, src_image, mode, mean, std):
        from torchvision.transforms import ToTensor, Normalize
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode

        arr = ToTensor()(image)
        pnormalize = PillowNormalize(mean=mean, std=std)
        tnormalize = Normalize(mean=mean, std=std)
        np.testing.assert_array_almost_equal(pnormalize(arr.numpy()), tnormalize(arr).numpy())

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'mode': [
            # 'I', 'I;16', 'F', '1', 'L', 'P',  # 1dim
            # 'LA',  # 2dim
            'RGB', 'YCbCr',  # 3dim
            # 'RGBA', 'CMYK',  # 4dim
        ],
        'mean': [
            (0.4850, 0.4560, 0.4060),
            (0.5, 0.5, 0.5),
            (0.0, 0.0, 0.0),
            (0.4850,),
            (0.5,),
            (0.0,),
        ],
        'std': [
            (0.2290, 0.2240, 0.2250),
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            (0.2290,),
            (0.5,),
            (1.0,),
        ]
    }))
    def test_maybe_normalize_3dim_upgrade(self, src_image, mode, mean, std):
        from torchvision.transforms import ToTensor, Normalize
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode

        arr = ToTensor()(image)
        arr = arr.unsqueeze(0).unsqueeze(0)
        pnormalize = PillowNormalize(mean=mean, std=std)
        tnormalize = Normalize(mean=mean, std=std)
        np.testing.assert_array_almost_equal(pnormalize(arr.numpy()), tnormalize(arr).numpy())

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'mode': [
            'F', '1', 'L', 'P',  # 1dim
            # 'LA',  # 2dim
            # 'RGB', 'YCbCr',  # 3dim
            # 'RGBA', 'CMYK',  # 4dim
        ],
        'mean': [
            (0.4850,),
            (0.5,),
            (0.0,),
        ],
        'std': [
            (0.2290,),
            (0.5,),
            (1.0,),
        ]
    }))
    def test_maybe_normalize_1dim(self, src_image, mode, mean, std):
        from torchvision.transforms import ToTensor, Normalize
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode

        arr = ToTensor()(image)
        pnormalize = PillowNormalize(mean=mean, std=std)
        tnormalize = Normalize(mean=mean, std=std)
        np.testing.assert_array_almost_equal(pnormalize(arr.numpy()), tnormalize(arr).numpy())

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'mode': [
            'F', '1', 'L', 'P',  # 1dim
            # 'LA',  # 2dim
            # 'RGB', 'YCbCr',  # 3dim
            # 'RGBA', 'CMYK',  # 4dim
        ],
        'mean': [
            0.4850,
            0.5,
            0.0,
        ],
        'std': [
            0.2290,
            0.5,
            1.0,
        ]
    }))
    def test_maybe_normalize_1dim_single_value(self, src_image, mode, mean, std):
        from torchvision.transforms import ToTensor, Normalize
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode

        arr = ToTensor()(image)
        pnormalize = PillowNormalize(mean=mean, std=std)
        tnormalize = Normalize(mean=mean, std=std)
        np.testing.assert_array_almost_equal(pnormalize(arr.numpy()), tnormalize(arr).numpy())

    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
    }))
    def test_maybe_normalize_invalid_image(self, src_image):
        image = Image.open(get_testfile(src_image))
        pnormalize = PillowNormalize(mean=0.5, std=0.5)
        with pytest.raises(TypeError):
            _ = pnormalize(image)

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'mode': [
            'I', 'I;16'
        ],
        'mean': [
            (0.4850, 0.4560, 0.4060),
            (0.5, 0.5, 0.5),
            (0.0, 0.0, 0.0),
            (0.4850,),
            (0.5,),
            (0.0,),
        ],
        'std': [
            (0.2290, 0.2240, 0.2250),
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            (0.2290,),
            (0.5,),
            (1.0,),
        ]
    }))
    def test_maybe_normalize_invalid_non_float(self, src_image, mode, mean, std):
        from torchvision.transforms import ToTensor
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode

        arr = ToTensor()(image)
        pnormalize = PillowNormalize(mean=mean, std=std)
        with pytest.raises(TypeError):
            _ = pnormalize(arr.numpy())

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'mode': [
            'F', '1', 'L', 'P',  # 1dim
        ],
        'mean': [
            (0.4850, 0.4560, 0.4060),
            (0.5, 0.5, 0.5),
            (0.0, 0.0, 0.0),
            (0.4850,),
            (0.5,),
            (0.0,),
        ],
        'std': [
            (0.2290, 0.2240, 0.2250),
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            (0.2290,),
            (0.5,),
            (1.0,),
        ]
    }))
    def test_maybe_normalize_invalid_CHW(self, src_image, mode, mean, std):
        from torchvision.transforms import ToTensor
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode

        arr = ToTensor()(image)
        assert arr.shape[0]
        arr = arr[0]
        pnormalize = PillowNormalize(mean=mean, std=std)
        with pytest.raises(ValueError):
            _ = pnormalize(arr.numpy())

    @pytest.mark.parametrize(['mean', 'std', 'repr_text'], [
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
         'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.229 0.224 0.225])'),
        ((0.485, 0.456, 0.406), [0.229, 0.224, 0.225],
         'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.229 0.224 0.225])'),
        ((0.485, 0.456, 0.406), (0.5, 0.5, 0.5), 'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.5 0.5 0.5])'),
        ((0.485, 0.456, 0.406), (1.0, 1.0, 1.0), 'PillowNormalize(mean=[0.485 0.456 0.406], std=[1. 1. 1.])'),
        ((0.485, 0.456, 0.406), (0.229,), 'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.229])'),
        ((0.485, 0.456, 0.406), (0.5,), 'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.5])'),
        ((0.485, 0.456, 0.406), (1.0,), 'PillowNormalize(mean=[0.485 0.456 0.406], std=[1.])'),
        ((0.485, 0.456, 0.406), 0.229, 'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.229])'),
        ((0.485, 0.456, 0.406), 0.5, 'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.5])'),
        ((0.485, 0.456, 0.406), 1.0, 'PillowNormalize(mean=[0.485 0.456 0.406], std=[1.])'),
        ([0.485, 0.456, 0.406], (0.229, 0.224, 0.225),
         'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.229 0.224 0.225])'),
        ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
         'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.229 0.224 0.225])'),
        ([0.485, 0.456, 0.406], (0.5, 0.5, 0.5), 'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.5 0.5 0.5])'),
        ([0.485, 0.456, 0.406], (1.0, 1.0, 1.0), 'PillowNormalize(mean=[0.485 0.456 0.406], std=[1. 1. 1.])'),
        ([0.485, 0.456, 0.406], (0.229,), 'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.229])'),
        ([0.485, 0.456, 0.406], (0.5,), 'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.5])'),
        ([0.485, 0.456, 0.406], (1.0,), 'PillowNormalize(mean=[0.485 0.456 0.406], std=[1.])'),
        ([0.485, 0.456, 0.406], 0.229, 'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.229])'),
        ([0.485, 0.456, 0.406], 0.5, 'PillowNormalize(mean=[0.485 0.456 0.406], std=[0.5])'),
        ([0.485, 0.456, 0.406], 1.0, 'PillowNormalize(mean=[0.485 0.456 0.406], std=[1.])'),
        ((0.5, 0.5, 0.5), (0.229, 0.224, 0.225), 'PillowNormalize(mean=[0.5 0.5 0.5], std=[0.229 0.224 0.225])'),
        ((0.5, 0.5, 0.5), [0.229, 0.224, 0.225], 'PillowNormalize(mean=[0.5 0.5 0.5], std=[0.229 0.224 0.225])'),
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 'PillowNormalize(mean=[0.5 0.5 0.5], std=[0.5 0.5 0.5])'),
        ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0), 'PillowNormalize(mean=[0.5 0.5 0.5], std=[1. 1. 1.])'),
        ((0.5, 0.5, 0.5), (0.229,), 'PillowNormalize(mean=[0.5 0.5 0.5], std=[0.229])'),
        ((0.5, 0.5, 0.5), (0.5,), 'PillowNormalize(mean=[0.5 0.5 0.5], std=[0.5])'),
        ((0.5, 0.5, 0.5), (1.0,), 'PillowNormalize(mean=[0.5 0.5 0.5], std=[1.])'),
        ((0.5, 0.5, 0.5), 0.229, 'PillowNormalize(mean=[0.5 0.5 0.5], std=[0.229])'),
        ((0.5, 0.5, 0.5), 0.5, 'PillowNormalize(mean=[0.5 0.5 0.5], std=[0.5])'),
        ((0.5, 0.5, 0.5), 1.0, 'PillowNormalize(mean=[0.5 0.5 0.5], std=[1.])'),
        ((0.0, 0.0, 0.0), (0.229, 0.224, 0.225), 'PillowNormalize(mean=[0. 0. 0.], std=[0.229 0.224 0.225])'),
        ((0.0, 0.0, 0.0), [0.229, 0.224, 0.225], 'PillowNormalize(mean=[0. 0. 0.], std=[0.229 0.224 0.225])'),
        ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 'PillowNormalize(mean=[0. 0. 0.], std=[0.5 0.5 0.5])'),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 'PillowNormalize(mean=[0. 0. 0.], std=[1. 1. 1.])'),
        ((0.0, 0.0, 0.0), (0.229,), 'PillowNormalize(mean=[0. 0. 0.], std=[0.229])'),
        ((0.0, 0.0, 0.0), (0.5,), 'PillowNormalize(mean=[0. 0. 0.], std=[0.5])'),
        ((0.0, 0.0, 0.0), (1.0,), 'PillowNormalize(mean=[0. 0. 0.], std=[1.])'),
        ((0.0, 0.0, 0.0), 0.229, 'PillowNormalize(mean=[0. 0. 0.], std=[0.229])'),
        ((0.0, 0.0, 0.0), 0.5, 'PillowNormalize(mean=[0. 0. 0.], std=[0.5])'),
        ((0.0, 0.0, 0.0), 1.0, 'PillowNormalize(mean=[0. 0. 0.], std=[1.])'),
        ((0.485,), (0.229, 0.224, 0.225), 'PillowNormalize(mean=[0.485], std=[0.229 0.224 0.225])'),
        ((0.485,), [0.229, 0.224, 0.225], 'PillowNormalize(mean=[0.485], std=[0.229 0.224 0.225])'),
        ((0.485,), (0.5, 0.5, 0.5), 'PillowNormalize(mean=[0.485], std=[0.5 0.5 0.5])'),
        ((0.485,), (1.0, 1.0, 1.0), 'PillowNormalize(mean=[0.485], std=[1. 1. 1.])'),
        ((0.485,), (0.229,), 'PillowNormalize(mean=[0.485], std=[0.229])'),
        ((0.485,), (0.5,), 'PillowNormalize(mean=[0.485], std=[0.5])'),
        ((0.485,), (1.0,), 'PillowNormalize(mean=[0.485], std=[1.])'),
        ((0.485,), 0.229, 'PillowNormalize(mean=[0.485], std=[0.229])'),
        ((0.485,), 0.5, 'PillowNormalize(mean=[0.485], std=[0.5])'),
        ((0.485,), 1.0, 'PillowNormalize(mean=[0.485], std=[1.])'),
        ((0.5,), (0.229, 0.224, 0.225), 'PillowNormalize(mean=[0.5], std=[0.229 0.224 0.225])'),
        ((0.5,), [0.229, 0.224, 0.225], 'PillowNormalize(mean=[0.5], std=[0.229 0.224 0.225])'),
        ((0.5,), (0.5, 0.5, 0.5), 'PillowNormalize(mean=[0.5], std=[0.5 0.5 0.5])'),
        ((0.5,), (1.0, 1.0, 1.0), 'PillowNormalize(mean=[0.5], std=[1. 1. 1.])'),
        ((0.5,), (0.229,), 'PillowNormalize(mean=[0.5], std=[0.229])'),
        ((0.5,), (0.5,), 'PillowNormalize(mean=[0.5], std=[0.5])'),
        ((0.5,), (1.0,), 'PillowNormalize(mean=[0.5], std=[1.])'),
        ((0.5,), 0.229, 'PillowNormalize(mean=[0.5], std=[0.229])'),
        ((0.5,), 0.5, 'PillowNormalize(mean=[0.5], std=[0.5])'),
        ((0.5,), 1.0, 'PillowNormalize(mean=[0.5], std=[1.])'),
        ((0.0,), (0.229, 0.224, 0.225), 'PillowNormalize(mean=[0.], std=[0.229 0.224 0.225])'),
        ((0.0,), [0.229, 0.224, 0.225], 'PillowNormalize(mean=[0.], std=[0.229 0.224 0.225])'),
        ((0.0,), (0.5, 0.5, 0.5), 'PillowNormalize(mean=[0.], std=[0.5 0.5 0.5])'),
        ((0.0,), (1.0, 1.0, 1.0), 'PillowNormalize(mean=[0.], std=[1. 1. 1.])'),
        ((0.0,), (0.229,), 'PillowNormalize(mean=[0.], std=[0.229])'),
        ((0.0,), (0.5,), 'PillowNormalize(mean=[0.], std=[0.5])'),
        ((0.0,), (1.0,), 'PillowNormalize(mean=[0.], std=[1.])'),
        ((0.0,), 0.229, 'PillowNormalize(mean=[0.], std=[0.229])'),
        ((0.0,), 0.5, 'PillowNormalize(mean=[0.], std=[0.5])'),
        ((0.0,), 1.0, 'PillowNormalize(mean=[0.], std=[1.])'),
        (0.485, (0.229, 0.224, 0.225), 'PillowNormalize(mean=[0.485], std=[0.229 0.224 0.225])'),
        (0.485, [0.229, 0.224, 0.225], 'PillowNormalize(mean=[0.485], std=[0.229 0.224 0.225])'),
        (0.485, (0.5, 0.5, 0.5), 'PillowNormalize(mean=[0.485], std=[0.5 0.5 0.5])'),
        (0.485, (1.0, 1.0, 1.0), 'PillowNormalize(mean=[0.485], std=[1. 1. 1.])'),
        (0.485, (0.229,), 'PillowNormalize(mean=[0.485], std=[0.229])'),
        (0.485, (0.5,), 'PillowNormalize(mean=[0.485], std=[0.5])'),
        (0.485, (1.0,), 'PillowNormalize(mean=[0.485], std=[1.])'),
        (0.485, 0.229, 'PillowNormalize(mean=[0.485], std=[0.229])'),
        (0.485, 0.5, 'PillowNormalize(mean=[0.485], std=[0.5])'),
        (0.485, 1.0, 'PillowNormalize(mean=[0.485], std=[1.])'),
        (0.5, (0.229, 0.224, 0.225), 'PillowNormalize(mean=[0.5], std=[0.229 0.224 0.225])'),
        (0.5, [0.229, 0.224, 0.225], 'PillowNormalize(mean=[0.5], std=[0.229 0.224 0.225])'),
        (0.5, (0.5, 0.5, 0.5), 'PillowNormalize(mean=[0.5], std=[0.5 0.5 0.5])'),
        (0.5, (1.0, 1.0, 1.0), 'PillowNormalize(mean=[0.5], std=[1. 1. 1.])'),
        (0.5, (0.229,), 'PillowNormalize(mean=[0.5], std=[0.229])'),
        (0.5, (0.5,), 'PillowNormalize(mean=[0.5], std=[0.5])'),
        (0.5, (1.0,), 'PillowNormalize(mean=[0.5], std=[1.])'),
        (0.5, 0.229, 'PillowNormalize(mean=[0.5], std=[0.229])'),
        (0.5, 0.5, 'PillowNormalize(mean=[0.5], std=[0.5])'),
        (0.5, 1.0, 'PillowNormalize(mean=[0.5], std=[1.])'),
        (0.0, (0.229, 0.224, 0.225), 'PillowNormalize(mean=[0.], std=[0.229 0.224 0.225])'),
        (0.0, [0.229, 0.224, 0.225], 'PillowNormalize(mean=[0.], std=[0.229 0.224 0.225])'),
        (0.0, (0.5, 0.5, 0.5), 'PillowNormalize(mean=[0.], std=[0.5 0.5 0.5])'),
        (0.0, (1.0, 1.0, 1.0), 'PillowNormalize(mean=[0.], std=[1. 1. 1.])'),
        (0.0, (0.229,), 'PillowNormalize(mean=[0.], std=[0.229])'),
        (0.0, (0.5,), 'PillowNormalize(mean=[0.], std=[0.5])'),
        (0.0, (1.0,), 'PillowNormalize(mean=[0.], std=[1.])'),
        (0.0, 0.229, 'PillowNormalize(mean=[0.], std=[0.229])'),
        (0.0, 0.5, 'PillowNormalize(mean=[0.], std=[0.5])'),
        (0.0, 1.0, 'PillowNormalize(mean=[0.], std=[1.])'),
    ])
    def test_normalize_repr(self, mean, std, repr_text):
        pnormalize = PillowNormalize(mean, std)
        assert repr(pnormalize) == repr_text

    def test_parse_pillow_transforms(self):
        assert parse_pillow_transforms(PillowCompose([
            PillowResize(size=384, interpolation=Image.BICUBIC, max_size=None, antialias=True),
            PillowCenterCrop(size=[384, 384]),
            PillowMaybeToTensor(),
            PillowNormalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000]),
        ])) == [
                   {'antialias': True,
                    'interpolation': 'bicubic',
                    'max_size': None,
                    'size': 384,
                    'type': 'resize'},
                   {'size': [384, 384], 'type': 'center_crop'},
                   {'type': 'maybe_to_tensor'},
                   {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'type': 'normalize'}
               ]

        assert parse_pillow_transforms(PillowCompose([
            PillowResize(size=384, interpolation=Image.BICUBIC, max_size=None, antialias=True),
            PillowCenterCrop(size=[384, 384]),
            PillowToTensor(),
            PillowNormalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000]),
        ])) == [
                   {'antialias': True,
                    'interpolation': 'bicubic',
                    'max_size': None,
                    'size': 384,
                    'type': 'resize'},
                   {'size': [384, 384], 'type': 'center_crop'},
                   {'type': 'to_tensor'},
                   {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'type': 'normalize'}
               ]

        assert parse_pillow_transforms(
            PillowResize(size=384, interpolation=Image.BICUBIC, max_size=None, antialias=True)) \
               == {'antialias': True,
                   'interpolation': 'bicubic',
                   'max_size': None,
                   'size': 384,
                   'type': 'resize'}
        assert parse_pillow_transforms(PillowCenterCrop(size=[384, 384])) == {'size': [384, 384], 'type': 'center_crop'}
        assert parse_pillow_transforms(PillowToTensor()) == {'type': 'to_tensor'}
        assert parse_pillow_transforms(
            PillowNormalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000])) \
               == {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'type': 'normalize'}

        with pytest.raises(TypeError):
            _ = parse_pillow_transforms(None)
        with pytest.raises(TypeError):
            _ = parse_pillow_transforms(23344)

    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
            'dori_640.png',
            'nian_640.png',
        ],
        'bg_color': [
            'white', 'green', 'blue', 'red', 'black',
        ]
    }))
    def test_convert_rgb(self, src_image, bg_color, image_diff):
        image = Image.open(get_testfile(src_image))
        pconvertrgb = PillowConvertRGB(force_background=bg_color)
        dst_image = pconvertrgb(image)
        assert dst_image.mode == 'RGB'

        assert image_diff(
            dst_image,
            load_image(image, force_background=bg_color, mode='RGB'),
            throw_exception=False
        ) < 1e-3

    def test_convert_rgb_invalid_input(self):
        pconvertrgb = PillowConvertRGB()
        with pytest.raises(TypeError):
            pconvertrgb(np.random.randn(1, 3, 384, 384))

    @pytest.mark.parametrize(['color', 'repr_text'], [
        (None, "PillowConvertRGB(force_background='white')"),
        ('white', "PillowConvertRGB(force_background='white')"),
        ('black', "PillowConvertRGB(force_background='black')"),
        ('red', "PillowConvertRGB(force_background='red')"),
        ('green', "PillowConvertRGB(force_background='green')"),
        ('blue', "PillowConvertRGB(force_background='blue')"),
    ])
    def test_convert_rgb_repr(self, color, repr_text):
        pconvertrgb = PillowConvertRGB() if color is None else PillowConvertRGB(color)
        assert repr(pconvertrgb) == repr_text

    @pytest.mark.parametrize(['color', 'json_data'], [
        ('white', {'type': 'convert_rgb'}),
        ('white', {'type': 'convert_rgb', 'force_background': 'white'}),
        ('black', {'type': 'convert_rgb', 'force_background': 'black'}),
        ('red', {'type': 'convert_rgb', 'force_background': 'red'}),
        ('green', {'type': 'convert_rgb', 'force_background': 'green'}),
        ('blue', {'type': 'convert_rgb', 'force_background': 'blue'}),
    ])
    def test_create_convert_rgb(self, color, json_data):
        pconvertrgb = create_pillow_transforms(json_data)
        assert isinstance(pconvertrgb, PillowConvertRGB)
        assert pconvertrgb.force_background == color

    @pytest.mark.parametrize(['color', 'json_data'], [
        (None, {'type': 'convert_rgb', 'force_background': 'white'}),
        ('white', {'type': 'convert_rgb', 'force_background': 'white'}),
        ('black', {'type': 'convert_rgb', 'force_background': 'black'}),
        ('red', {'type': 'convert_rgb', 'force_background': 'red'}),
        ('green', {'type': 'convert_rgb', 'force_background': 'green'}),
        ('blue', {'type': 'convert_rgb', 'force_background': 'blue'}),
    ])
    def test_parse_convert_rgb(self, color, json_data):
        pconvertrgb = PillowConvertRGB() if color is None else PillowConvertRGB(color)
        assert parse_pillow_transforms(pconvertrgb) == json_data

    @pytest.mark.parametrize(*tmatrix({
        'seed': list(range(10)),
        'rescale_factor': [1 / 255, 1 / 254, 1 / 256, 1 / 127, 0.5, 0.1, 2, 10, 255, 254, 256],
    }))
    def test_rescale(self, seed, rescale_factor):
        np.random.seed(seed)
        arr = np.random.randn(3, 384, 384)
        prescale = PillowRescale(rescale_factor=rescale_factor)
        np.testing.assert_array_almost_equal(arr * rescale_factor, prescale(arr))

    @pytest.mark.parametrize(['rescale_factor', 'repr_text'], [
        (1 / 255, 'PillowRescale(rescale_factor=0.003921569)'),
        (1 / 254, 'PillowRescale(rescale_factor=0.003937008)'),
        (1 / 256, 'PillowRescale(rescale_factor=0.00390625)'),
        (1 / 127, 'PillowRescale(rescale_factor=0.007874016)'),
        (1 / 2, 'PillowRescale(rescale_factor=0.5)'),
        (1 / 10, 'PillowRescale(rescale_factor=0.1)'),
        (2, 'PillowRescale(rescale_factor=2.0)'),
        (10, 'PillowRescale(rescale_factor=10.0)'),
        (255, 'PillowRescale(rescale_factor=255.0)'),
        (254, 'PillowRescale(rescale_factor=254.0)'),
        (256, 'PillowRescale(rescale_factor=256.0)'),
    ])
    def test_rescale_repr(self, rescale_factor, repr_text):
        prescale = PillowRescale(rescale_factor)
        assert repr(prescale) == repr_text

    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
            'dori_640.png',
            'nian_640.png',
        ],
    }))
    def test_rescale_input_invalid(self, src_image):
        prescale = PillowRescale(1 / 255)
        image = Image.open(get_testfile(src_image))
        with pytest.raises(TypeError):
            _ = prescale(image)

    @pytest.mark.parametrize(['rescale_factor', 'json_data'], [
        (1 / 255, {'type': 'rescale', 'rescale_factor': 0.003921568859368563}),
        (1 / 254, {'type': 'rescale', 'rescale_factor': 0.003937007859349251}),
        (1 / 256, {'type': 'rescale', 'rescale_factor': 0.00390625}),
        (1 / 127, {'type': 'rescale', 'rescale_factor': 0.007874015718698502}),
        (1 / 2, {'type': 'rescale', 'rescale_factor': 0.5}),
        (1 / 10, {'type': 'rescale', 'rescale_factor': 0.10000000149011612}),
        (2, {'type': 'rescale', 'rescale_factor': 2.0}),
        (10, {'type': 'rescale', 'rescale_factor': 10.0}),
        (255, {'type': 'rescale', 'rescale_factor': 255.0}),
        (254, {'type': 'rescale', 'rescale_factor': 254.0}),
        (256, {'type': 'rescale', 'rescale_factor': 256.0}),
    ])
    def test_rescale_parse(self, rescale_factor, json_data):
        prescale = PillowRescale(rescale_factor)
        assert parse_pillow_transforms(prescale) == pytest.approx(json_data, abs=1e-5)

    @pytest.mark.parametrize(['rescale_factor', 'json_data'], [
        (1 / 255, {'type': 'rescale', 'rescale_factor': 0.003921568859368563}),
        (1 / 254, {'type': 'rescale', 'rescale_factor': 0.003937007859349251}),
        (1 / 256, {'type': 'rescale', 'rescale_factor': 0.00390625}),
        (1 / 127, {'type': 'rescale', 'rescale_factor': 0.007874015718698502}),
        (1 / 2, {'type': 'rescale', 'rescale_factor': 0.5}),
        (1 / 10, {'type': 'rescale', 'rescale_factor': 0.10000000149011612}),
        (2, {'type': 'rescale', 'rescale_factor': 2.0}),
        (10, {'type': 'rescale', 'rescale_factor': 10.0}),
        (255, {'type': 'rescale', 'rescale_factor': 255.0}),
        (254, {'type': 'rescale', 'rescale_factor': 254.0}),
        (256, {'type': 'rescale', 'rescale_factor': 256.0}),
    ])
    def test_rescale_create(self, rescale_factor, json_data):
        prescale = create_pillow_transforms(json_data)
        assert isinstance(prescale, PillowRescale)
        assert prescale.rescale_factor == pytest.approx(rescale_factor, abs=1e-5)

    @pytest.mark.parametrize(['size', 'background_color', 'interpolation', 'repr_text'], [
        [(384, 512), 'white', 5, 'PillowPadToSize(size=(384, 512), interpolation=hamming, background_color=white)'],
        [(512, 384), 'white', 4, 'PillowPadToSize(size=(512, 384), interpolation=box, background_color=white)'],
        [(768, 512), 'white', 3, 'PillowPadToSize(size=(768, 512), interpolation=bicubic, background_color=white)'],
        [(512, 512), 'white', 2, 'PillowPadToSize(size=(512, 512), interpolation=bilinear, background_color=white)'],
        [(512, 768), 'white', 1, 'PillowPadToSize(size=(512, 768), interpolation=lanczos, background_color=white)'],
        [(512, 768), 'white', 0, 'PillowPadToSize(size=(512, 768), interpolation=nearest, background_color=white)'],
        [(512, 384), 'red', 5, 'PillowPadToSize(size=(512, 384), interpolation=hamming, background_color=red)'],
        [(768, 512), 'red', 4, 'PillowPadToSize(size=(768, 512), interpolation=box, background_color=red)'],
        [(512, 512), 'red', 3, 'PillowPadToSize(size=(512, 512), interpolation=bicubic, background_color=red)'],
        [(384, 512), 'red', 2, 'PillowPadToSize(size=(384, 512), interpolation=bilinear, background_color=red)'],
        [(384, 512), 'red', 1, 'PillowPadToSize(size=(384, 512), interpolation=lanczos, background_color=red)'],
        [(768, 512), 'red', 0, 'PillowPadToSize(size=(768, 512), interpolation=nearest, background_color=red)'],
        [(512, 768), 'green', 5, 'PillowPadToSize(size=(512, 768), interpolation=hamming, background_color=green)'],
        [(512, 512), 'green', 4, 'PillowPadToSize(size=(512, 512), interpolation=box, background_color=green)'],
        [(512, 384), 'green', 3, 'PillowPadToSize(size=(512, 384), interpolation=bicubic, background_color=green)'],
        [(768, 512), 'green', 2, 'PillowPadToSize(size=(768, 512), interpolation=bilinear, background_color=green)'],
        [(512, 384), 'green', 1, 'PillowPadToSize(size=(512, 384), interpolation=lanczos, background_color=green)'],
        [(384, 512), 'green', 0, 'PillowPadToSize(size=(384, 512), interpolation=nearest, background_color=green)'],
        [(512, 512), 'gray', 5, 'PillowPadToSize(size=(512, 512), interpolation=hamming, background_color=gray)'],
        [(512, 768), 'gray', 4, 'PillowPadToSize(size=(512, 768), interpolation=box, background_color=gray)'],
        [(384, 512), 'gray', 3, 'PillowPadToSize(size=(384, 512), interpolation=bicubic, background_color=gray)'],
        [(512, 384), 'gray', 2, 'PillowPadToSize(size=(512, 384), interpolation=bilinear, background_color=gray)'],
        [(768, 512), 'gray', 1, 'PillowPadToSize(size=(768, 512), interpolation=lanczos, background_color=gray)'],
        [(512, 512), 'gray', 0, 'PillowPadToSize(size=(512, 512), interpolation=nearest, background_color=gray)'],
        [(768, 512), 'blue', 5, 'PillowPadToSize(size=(768, 512), interpolation=hamming, background_color=blue)'],
        [(384, 512), 'blue', 4, 'PillowPadToSize(size=(384, 512), interpolation=box, background_color=blue)'],
        [(512, 768), 'blue', 3, 'PillowPadToSize(size=(512, 768), interpolation=bicubic, background_color=blue)'],
        [(512, 768), 'blue', 2, 'PillowPadToSize(size=(512, 768), interpolation=bilinear, background_color=blue)'],
        [(512, 512), 'blue', 1, 'PillowPadToSize(size=(512, 512), interpolation=lanczos, background_color=blue)'],
        [(512, 384), 'blue', 0, 'PillowPadToSize(size=(512, 384), interpolation=nearest, background_color=blue)'],
        [(512, 768), 'black', 5, 'PillowPadToSize(size=(512, 768), interpolation=hamming, background_color=black)'],
        [(512, 512), 'black', 4, 'PillowPadToSize(size=(512, 512), interpolation=box, background_color=black)'],
        [(768, 512), 'black', 3, 'PillowPadToSize(size=(768, 512), interpolation=bicubic, background_color=black)'],
        [(512, 384), 'black', 2, 'PillowPadToSize(size=(512, 384), interpolation=bilinear, background_color=black)'],
        [(384, 512), 'black', 1, 'PillowPadToSize(size=(384, 512), interpolation=lanczos, background_color=black)'],
        [(512, 768), 'black', 0, 'PillowPadToSize(size=(512, 768), interpolation=nearest, background_color=black)'],
        [(512, 768), 'red', 3, 'PillowPadToSize(size=(512, 768), interpolation=bicubic, background_color=red)'],
        [384, (255, 255, 255), 5,
         'PillowPadToSize(size=(384, 384), interpolation=hamming, background_color=(255, 255, 255))'],
        [512, (255, 255, 255), 4,
         'PillowPadToSize(size=(512, 512), interpolation=box, background_color=(255, 255, 255))'],
        [768, (255, 255, 255), 3,
         'PillowPadToSize(size=(768, 768), interpolation=bicubic, background_color=(255, 255, 255))'],
        [512, (255, 255, 255), 2,
         'PillowPadToSize(size=(512, 512), interpolation=bilinear, background_color=(255, 255, 255))'],
        [768, (255, 255, 255), 1,
         'PillowPadToSize(size=(768, 768), interpolation=lanczos, background_color=(255, 255, 255))'],
        [384, (255, 255, 255), 0,
         'PillowPadToSize(size=(384, 384), interpolation=nearest, background_color=(255, 255, 255))'],
        [512, (255, 0, 0, 128), 5,
         'PillowPadToSize(size=(512, 512), interpolation=hamming, background_color=(255, 0, 0, 128))'],
        [384, (255, 0, 0, 128), 4,
         'PillowPadToSize(size=(384, 384), interpolation=box, background_color=(255, 0, 0, 128))'],
        [384, (255, 0, 0, 128), 3,
         'PillowPadToSize(size=(384, 384), interpolation=bicubic, background_color=(255, 0, 0, 128))'],
        [768, (255, 0, 0, 128), 2,
         'PillowPadToSize(size=(768, 768), interpolation=bilinear, background_color=(255, 0, 0, 128))'],
        [512, (255, 0, 0, 128), 1,
         'PillowPadToSize(size=(512, 512), interpolation=lanczos, background_color=(255, 0, 0, 128))'],
        [768, (255, 0, 0, 128), 0,
         'PillowPadToSize(size=(768, 768), interpolation=nearest, background_color=(255, 0, 0, 128))'],
        [768, (255, 0, 0), 5, 'PillowPadToSize(size=(768, 768), interpolation=hamming, background_color=(255, 0, 0))'],
        [768, (255, 0, 0), 4, 'PillowPadToSize(size=(768, 768), interpolation=box, background_color=(255, 0, 0))'],
        [512, (255, 0, 0), 3, 'PillowPadToSize(size=(512, 512), interpolation=bicubic, background_color=(255, 0, 0))'],
        [384, (255, 0, 0), 2, 'PillowPadToSize(size=(384, 384), interpolation=bilinear, background_color=(255, 0, 0))'],
        [384, (255, 0, 0), 1, 'PillowPadToSize(size=(384, 384), interpolation=lanczos, background_color=(255, 0, 0))'],
        [512, (255, 0, 0), 0, 'PillowPadToSize(size=(512, 512), interpolation=nearest, background_color=(255, 0, 0))'],
        [512, (0, 255, 0, 128), 5,
         'PillowPadToSize(size=(512, 512), interpolation=hamming, background_color=(0, 255, 0, 128))'],
        [384, (0, 255, 0, 128), 4,
         'PillowPadToSize(size=(384, 384), interpolation=box, background_color=(0, 255, 0, 128))'],
        [768, (0, 255, 0, 128), 3,
         'PillowPadToSize(size=(768, 768), interpolation=bicubic, background_color=(0, 255, 0, 128))'],
        [384, (0, 255, 0, 128), 2,
         'PillowPadToSize(size=(384, 384), interpolation=bilinear, background_color=(0, 255, 0, 128))'],
        [384, (0, 255, 0, 128), 1,
         'PillowPadToSize(size=(384, 384), interpolation=lanczos, background_color=(0, 255, 0, 128))'],
        [384, (0, 255, 0, 128), 0,
         'PillowPadToSize(size=(384, 384), interpolation=nearest, background_color=(0, 255, 0, 128))'],
        [512, (0, 255, 0), 5, 'PillowPadToSize(size=(512, 512), interpolation=hamming, background_color=(0, 255, 0))'],
        [768, (0, 255, 0), 4, 'PillowPadToSize(size=(768, 768), interpolation=box, background_color=(0, 255, 0))'],
        [384, (0, 255, 0), 3, 'PillowPadToSize(size=(384, 384), interpolation=bicubic, background_color=(0, 255, 0))'],
        [384, (0, 255, 0), 2, 'PillowPadToSize(size=(384, 384), interpolation=bilinear, background_color=(0, 255, 0))'],
        [768, (0, 255, 0), 1, 'PillowPadToSize(size=(768, 768), interpolation=lanczos, background_color=(0, 255, 0))'],
        [512, (0, 255, 0), 0, 'PillowPadToSize(size=(512, 512), interpolation=nearest, background_color=(0, 255, 0))'],
        [384, (0, 0, 255, 128), 5,
         'PillowPadToSize(size=(384, 384), interpolation=hamming, background_color=(0, 0, 255, 128))'],
        [512, (0, 0, 255, 128), 4,
         'PillowPadToSize(size=(512, 512), interpolation=box, background_color=(0, 0, 255, 128))'],
        [768, (0, 0, 255, 128), 3,
         'PillowPadToSize(size=(768, 768), interpolation=bicubic, background_color=(0, 0, 255, 128))'],
        [768, (0, 0, 255, 128), 2,
         'PillowPadToSize(size=(768, 768), interpolation=bilinear, background_color=(0, 0, 255, 128))'],
        [512, (0, 0, 255, 128), 1,
         'PillowPadToSize(size=(512, 512), interpolation=lanczos, background_color=(0, 0, 255, 128))'],
        [512, (0, 0, 255, 128), 0,
         'PillowPadToSize(size=(512, 512), interpolation=nearest, background_color=(0, 0, 255, 128))'],
        [384, (0, 0, 255), 5, 'PillowPadToSize(size=(384, 384), interpolation=hamming, background_color=(0, 0, 255))'],
        [768, (0, 0, 255), 4, 'PillowPadToSize(size=(768, 768), interpolation=box, background_color=(0, 0, 255))'],
        [512, (0, 0, 255), 3, 'PillowPadToSize(size=(512, 512), interpolation=bicubic, background_color=(0, 0, 255))'],
        [512, (0, 0, 255), 2, 'PillowPadToSize(size=(512, 512), interpolation=bilinear, background_color=(0, 0, 255))'],
        [384, (0, 0, 255), 1, 'PillowPadToSize(size=(384, 384), interpolation=lanczos, background_color=(0, 0, 255))'],
        [384, (0, 0, 255), 0, 'PillowPadToSize(size=(384, 384), interpolation=nearest, background_color=(0, 0, 255))'],
        [384, (0, 0, 0), 5, 'PillowPadToSize(size=(384, 384), interpolation=hamming, background_color=(0, 0, 0))'],
        [768, (0, 0, 0), 4, 'PillowPadToSize(size=(768, 768), interpolation=box, background_color=(0, 0, 0))'],
        [512, (0, 0, 0), 3, 'PillowPadToSize(size=(512, 512), interpolation=bicubic, background_color=(0, 0, 0))'],
        [768, (0, 0, 0), 2, 'PillowPadToSize(size=(768, 768), interpolation=bilinear, background_color=(0, 0, 0))'],
        [768, (0, 0, 0), 1, 'PillowPadToSize(size=(768, 768), interpolation=lanczos, background_color=(0, 0, 0))'],
        [768, (0, 0, 0), 0, 'PillowPadToSize(size=(768, 768), interpolation=nearest, background_color=(0, 0, 0))'],
    ])
    def test_pad_to_size_repr(self, size, background_color, interpolation, repr_text):
        assert repr(PillowPadToSize(size=size, background_color=background_color, interpolation=interpolation)) == \
               repr_text

    @pytest.mark.parametrize(['size', 'background_color', 'interpolation', 'json_'], [
        [(384, 512), 'white', 5,
         {'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'white', 'interpolation': 'hamming'}],
        [(512, 384), 'white', 4,
         {'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'white', 'interpolation': 'box'}],
        [(768, 512), 'white', 3,
         {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'white', 'interpolation': 'bicubic'}],
        [(512, 512), 'white', 2,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'white', 'interpolation': 'bilinear'}],
        [(512, 768), 'white', 1,
         {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'white', 'interpolation': 'lanczos'}],
        [(512, 768), 'white', 0,
         {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'white', 'interpolation': 'nearest'}],
        [(512, 384), 'red', 5,
         {'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'red', 'interpolation': 'hamming'}],
        [(768, 512), 'red', 4,
         {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'red', 'interpolation': 'box'}],
        [(512, 512), 'red', 3,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'red', 'interpolation': 'bicubic'}],
        [(384, 512), 'red', 2,
         {'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'red', 'interpolation': 'bilinear'}],
        [(384, 512), 'red', 1,
         {'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'red', 'interpolation': 'lanczos'}],
        [(768, 512), 'red', 0,
         {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'red', 'interpolation': 'nearest'}],
        [(512, 768), 'green', 5,
         {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'green', 'interpolation': 'hamming'}],
        [(512, 512), 'green', 4,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'green', 'interpolation': 'box'}],
        [(512, 384), 'green', 3,
         {'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'green', 'interpolation': 'bicubic'}],
        [(768, 512), 'green', 2,
         {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'green', 'interpolation': 'bilinear'}],
        [(512, 384), 'green', 1,
         {'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'green', 'interpolation': 'lanczos'}],
        [(384, 512), 'green', 0,
         {'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'green', 'interpolation': 'nearest'}],
        [(512, 512), 'gray', 5,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'gray', 'interpolation': 'hamming'}],
        [(512, 768), 'gray', 4,
         {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'gray', 'interpolation': 'box'}],
        [(384, 512), 'gray', 3,
         {'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'gray', 'interpolation': 'bicubic'}],
        [(512, 384), 'gray', 2,
         {'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'gray', 'interpolation': 'bilinear'}],
        [(768, 512), 'gray', 1,
         {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'gray', 'interpolation': 'lanczos'}],
        [(512, 512), 'gray', 0,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'gray', 'interpolation': 'nearest'}],
        [(768, 512), 'blue', 5,
         {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'blue', 'interpolation': 'hamming'}],
        [(384, 512), 'blue', 4,
         {'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'blue', 'interpolation': 'box'}],
        [(512, 768), 'blue', 3,
         {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'blue', 'interpolation': 'bicubic'}],
        [(512, 768), 'blue', 2,
         {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'blue', 'interpolation': 'bilinear'}],
        [(512, 512), 'blue', 1,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'blue', 'interpolation': 'lanczos'}],
        [(512, 384), 'blue', 0,
         {'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'blue', 'interpolation': 'nearest'}],
        [(512, 768), 'black', 5,
         {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'black', 'interpolation': 'hamming'}],
        [(512, 512), 'black', 4,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'black', 'interpolation': 'box'}],
        [(768, 512), 'black', 3,
         {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'black', 'interpolation': 'bicubic'}],
        [(512, 384), 'black', 2,
         {'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'black', 'interpolation': 'bilinear'}],
        [(384, 512), 'black', 1,
         {'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'black', 'interpolation': 'lanczos'}],
        [(512, 768), 'black', 0,
         {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'black', 'interpolation': 'nearest'}],
        [(512, 768), 'red', 3,
         {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'red', 'interpolation': 'bicubic'}],
        [384, (255, 255, 255), 5,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 255, 255], 'interpolation': 'hamming'}],
        [512, (255, 255, 255), 4,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 255, 255], 'interpolation': 'box'}],
        [768, (255, 255, 255), 3,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [255, 255, 255], 'interpolation': 'bicubic'}],
        [512, (255, 255, 255), 2,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 255, 255], 'interpolation': 'bilinear'}],
        [768, (255, 255, 255), 1,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [255, 255, 255], 'interpolation': 'lanczos'}],
        [384, (255, 255, 255), 0,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 255, 255], 'interpolation': 'nearest'}],
        [512, (255, 0, 0, 128), 5,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 0, 0, 128], 'interpolation': 'hamming'}],
        [384, (255, 0, 0, 128), 4,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 0, 0, 128], 'interpolation': 'box'}],
        [384, (255, 0, 0, 128), 3,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 0, 0, 128], 'interpolation': 'bicubic'}],
        [768, (255, 0, 0, 128), 2, {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [255, 0, 0, 128],
                                    'interpolation': 'bilinear'}],
        [512, (255, 0, 0, 128), 1,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 0, 0, 128], 'interpolation': 'lanczos'}],
        [768, (255, 0, 0, 128), 0,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [255, 0, 0, 128], 'interpolation': 'nearest'}],
        [768, (255, 0, 0), 5,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [255, 0, 0], 'interpolation': 'hamming'}],
        [768, (255, 0, 0), 4,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [255, 0, 0], 'interpolation': 'box'}],
        [512, (255, 0, 0), 3,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 0, 0], 'interpolation': 'bicubic'}],
        [384, (255, 0, 0), 2,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 0, 0], 'interpolation': 'bilinear'}],
        [384, (255, 0, 0), 1,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 0, 0], 'interpolation': 'lanczos'}],
        [512, (255, 0, 0), 0,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 0, 0], 'interpolation': 'nearest'}],
        [512, (0, 255, 0, 128), 5,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 255, 0, 128], 'interpolation': 'hamming'}],
        [384, (0, 255, 0, 128), 4,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 255, 0, 128], 'interpolation': 'box'}],
        [768, (0, 255, 0, 128), 3,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 255, 0, 128], 'interpolation': 'bicubic'}],
        [384, (0, 255, 0, 128), 2, {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 255, 0, 128],
                                    'interpolation': 'bilinear'}],
        [384, (0, 255, 0, 128), 1,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 255, 0, 128], 'interpolation': 'lanczos'}],
        [384, (0, 255, 0, 128), 0,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 255, 0, 128], 'interpolation': 'nearest'}],
        [512, (0, 255, 0), 5,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 255, 0], 'interpolation': 'hamming'}],
        [768, (0, 255, 0), 4,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 255, 0], 'interpolation': 'box'}],
        [384, (0, 255, 0), 3,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 255, 0], 'interpolation': 'bicubic'}],
        [384, (0, 255, 0), 2,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 255, 0], 'interpolation': 'bilinear'}],
        [768, (0, 255, 0), 1,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 255, 0], 'interpolation': 'lanczos'}],
        [512, (0, 255, 0), 0,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 255, 0], 'interpolation': 'nearest'}],
        [384, (0, 0, 255, 128), 5,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 0, 255, 128], 'interpolation': 'hamming'}],
        [512, (0, 0, 255, 128), 4,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 255, 128], 'interpolation': 'box'}],
        [768, (0, 0, 255, 128), 3,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 255, 128], 'interpolation': 'bicubic'}],
        [768, (0, 0, 255, 128), 2, {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 255, 128],
                                    'interpolation': 'bilinear'}],
        [512, (0, 0, 255, 128), 1,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 255, 128], 'interpolation': 'lanczos'}],
        [512, (0, 0, 255, 128), 0,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 255, 128], 'interpolation': 'nearest'}],
        [384, (0, 0, 255), 5,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 0, 255], 'interpolation': 'hamming'}],
        [768, (0, 0, 255), 4,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 255], 'interpolation': 'box'}],
        [512, (0, 0, 255), 3,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 255], 'interpolation': 'bicubic'}],
        [512, (0, 0, 255), 2,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 255], 'interpolation': 'bilinear'}],
        [384, (0, 0, 255), 1,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 0, 255], 'interpolation': 'lanczos'}],
        [384, (0, 0, 255), 0,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 0, 255], 'interpolation': 'nearest'}],
        [384, (0, 0, 0), 5,
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 0, 0], 'interpolation': 'hamming'}],
        [768, (0, 0, 0), 4,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 0], 'interpolation': 'box'}],
        [512, (0, 0, 0), 3,
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 0], 'interpolation': 'bicubic'}],
        [768, (0, 0, 0), 2,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 0], 'interpolation': 'bilinear'}],
        [768, (0, 0, 0), 1,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 0], 'interpolation': 'lanczos'}],
        [768, (0, 0, 0), 0,
         {'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 0], 'interpolation': 'nearest'}],
    ])
    def test_pad_to_size_parse_json(self, size, background_color, interpolation, json_):
        assert parse_pillow_transforms(PillowPadToSize(size=size, background_color=background_color,
                                                       interpolation=interpolation)) == json_

    @pytest.mark.parametrize(['json_', 'size', 'background_color', 'interpolation'], [
        [{'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'white', 'interpolation': 'hamming'},
         (384, 512), 'white', 5],
        [{'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'white', 'interpolation': 'box'}, (512, 384),
         'white', 4],
        [{'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'white', 'interpolation': 'bicubic'},
         (768, 512), 'white', 3],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'white', 'interpolation': 'bilinear'},
         (512, 512), 'white', 2],
        [{'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'white', 'interpolation': 'lanczos'},
         (512, 768), 'white', 1],
        [{'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'white', 'interpolation': 'nearest'},
         (512, 768), 'white', 0],
        [{'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'red', 'interpolation': 'hamming'}, (512, 384),
         'red', 5],
        [{'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'red', 'interpolation': 'box'}, (768, 512),
         'red', 4],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'red', 'interpolation': 'bicubic'}, (512, 512),
         'red', 3],
        [{'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'red', 'interpolation': 'bilinear'},
         (384, 512), 'red', 2],
        [{'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'red', 'interpolation': 'lanczos'}, (384, 512),
         'red', 1],
        [{'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'red', 'interpolation': 'nearest'}, (768, 512),
         'red', 0],
        [{'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'green', 'interpolation': 'hamming'},
         (512, 768), 'green', 5],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'green', 'interpolation': 'box'}, (512, 512),
         'green', 4],
        [{'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'green', 'interpolation': 'bicubic'},
         (512, 384), 'green', 3],
        [{'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'green', 'interpolation': 'bilinear'},
         (768, 512), 'green', 2],
        [{'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'green', 'interpolation': 'lanczos'},
         (512, 384), 'green', 1],
        [{'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'green', 'interpolation': 'nearest'},
         (384, 512), 'green', 0],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'gray', 'interpolation': 'hamming'},
         (512, 512), 'gray', 5],
        [{'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'gray', 'interpolation': 'box'}, (512, 768),
         'gray', 4],
        [{'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'gray', 'interpolation': 'bicubic'},
         (384, 512), 'gray', 3],
        [{'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'gray', 'interpolation': 'bilinear'},
         (512, 384), 'gray', 2],
        [{'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'gray', 'interpolation': 'lanczos'},
         (768, 512), 'gray', 1],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'gray', 'interpolation': 'nearest'},
         (512, 512), 'gray', 0],
        [{'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'blue', 'interpolation': 'hamming'},
         (768, 512), 'blue', 5],
        [{'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'blue', 'interpolation': 'box'}, (384, 512),
         'blue', 4],
        [{'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'blue', 'interpolation': 'bicubic'},
         (512, 768), 'blue', 3],
        [{'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'blue', 'interpolation': 'bilinear'},
         (512, 768), 'blue', 2],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'blue', 'interpolation': 'lanczos'},
         (512, 512), 'blue', 1],
        [{'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'blue', 'interpolation': 'nearest'},
         (512, 384), 'blue', 0],
        [{'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'black', 'interpolation': 'hamming'},
         (512, 768), 'black', 5],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'black', 'interpolation': 'box'}, (512, 512),
         'black', 4],
        [{'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'black', 'interpolation': 'bicubic'},
         (768, 512), 'black', 3],
        [{'type': 'pad_to_size', 'size': [512, 384], 'background_color': 'black', 'interpolation': 'bilinear'},
         (512, 384), 'black', 2],
        [{'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'black', 'interpolation': 'lanczos'},
         (384, 512), 'black', 1],
        [{'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'black', 'interpolation': 'nearest'},
         (512, 768), 'black', 0],
        [{'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'red', 'interpolation': 'bicubic'}, (512, 768),
         'red', 3],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 255, 255], 'interpolation': 'hamming'},
         (384, 384), (255, 255, 255), 5],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 255, 255], 'interpolation': 'box'},
         (512, 512), (255, 255, 255), 4],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [255, 255, 255], 'interpolation': 'bicubic'},
         (768, 768), (255, 255, 255), 3],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 255, 255], 'interpolation': 'bilinear'},
         (512, 512), (255, 255, 255), 2],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [255, 255, 255], 'interpolation': 'lanczos'},
         (768, 768), (255, 255, 255), 1],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 255, 255], 'interpolation': 'nearest'},
         (384, 384), (255, 255, 255), 0],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 0, 0, 128], 'interpolation': 'hamming'},
         (512, 512), (255, 0, 0, 128), 5],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 0, 0, 128], 'interpolation': 'box'},
         (384, 384), (255, 0, 0, 128), 4],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 0, 0, 128], 'interpolation': 'bicubic'},
         (384, 384), (255, 0, 0, 128), 3],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [255, 0, 0, 128], 'interpolation': 'bilinear'},
         (768, 768), (255, 0, 0, 128), 2],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 0, 0, 128], 'interpolation': 'lanczos'},
         (512, 512), (255, 0, 0, 128), 1],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [255, 0, 0, 128], 'interpolation': 'nearest'},
         (768, 768), (255, 0, 0, 128), 0],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [255, 0, 0], 'interpolation': 'hamming'},
         (768, 768), (255, 0, 0), 5],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [255, 0, 0], 'interpolation': 'box'},
         (768, 768), (255, 0, 0), 4],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 0, 0], 'interpolation': 'bicubic'},
         (512, 512), (255, 0, 0), 3],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 0, 0], 'interpolation': 'bilinear'},
         (384, 384), (255, 0, 0), 2],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 0, 0], 'interpolation': 'lanczos'},
         (384, 384), (255, 0, 0), 1],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 0, 0], 'interpolation': 'nearest'},
         (512, 512), (255, 0, 0), 0],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 255, 0, 128], 'interpolation': 'hamming'},
         (512, 512), (0, 255, 0, 128), 5],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 255, 0, 128], 'interpolation': 'box'},
         (384, 384), (0, 255, 0, 128), 4],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 255, 0, 128], 'interpolation': 'bicubic'},
         (768, 768), (0, 255, 0, 128), 3],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 255, 0, 128], 'interpolation': 'bilinear'},
         (384, 384), (0, 255, 0, 128), 2],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 255, 0, 128], 'interpolation': 'lanczos'},
         (384, 384), (0, 255, 0, 128), 1],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 255, 0, 128], 'interpolation': 'nearest'},
         (384, 384), (0, 255, 0, 128), 0],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 255, 0], 'interpolation': 'hamming'},
         (512, 512), (0, 255, 0), 5],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 255, 0], 'interpolation': 'box'},
         (768, 768), (0, 255, 0), 4],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 255, 0], 'interpolation': 'bicubic'},
         (384, 384), (0, 255, 0), 3],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 255, 0], 'interpolation': 'bilinear'},
         (384, 384), (0, 255, 0), 2],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 255, 0], 'interpolation': 'lanczos'},
         (768, 768), (0, 255, 0), 1],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 255, 0], 'interpolation': 'nearest'},
         (512, 512), (0, 255, 0), 0],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 0, 255, 128], 'interpolation': 'hamming'},
         (384, 384), (0, 0, 255, 128), 5],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 255, 128], 'interpolation': 'box'},
         (512, 512), (0, 0, 255, 128), 4],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 255, 128], 'interpolation': 'bicubic'},
         (768, 768), (0, 0, 255, 128), 3],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 255, 128], 'interpolation': 'bilinear'},
         (768, 768), (0, 0, 255, 128), 2],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 255, 128], 'interpolation': 'lanczos'},
         (512, 512), (0, 0, 255, 128), 1],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 255, 128], 'interpolation': 'nearest'},
         (512, 512), (0, 0, 255, 128), 0],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 0, 255], 'interpolation': 'hamming'},
         (384, 384), (0, 0, 255), 5],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 255], 'interpolation': 'box'},
         (768, 768), (0, 0, 255), 4],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 255], 'interpolation': 'bicubic'},
         (512, 512), (0, 0, 255), 3],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 255], 'interpolation': 'bilinear'},
         (512, 512), (0, 0, 255), 2],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 0, 255], 'interpolation': 'lanczos'},
         (384, 384), (0, 0, 255), 1],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 0, 255], 'interpolation': 'nearest'},
         (384, 384), (0, 0, 255), 0],
        [{'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 0, 0], 'interpolation': 'hamming'},
         (384, 384), (0, 0, 0), 5],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 0], 'interpolation': 'box'}, (768, 768),
         (0, 0, 0), 4],
        [{'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 0], 'interpolation': 'bicubic'},
         (512, 512), (0, 0, 0), 3],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 0], 'interpolation': 'bilinear'},
         (768, 768), (0, 0, 0), 2],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 0], 'interpolation': 'lanczos'},
         (768, 768), (0, 0, 0), 1],
        [{'type': 'pad_to_size', 'size': [768, 768], 'background_color': [0, 0, 0], 'interpolation': 'nearest'},
         (768, 768), (0, 0, 0), 0],

    ])
    def test_pad_to_size_created_by_json(self, size, background_color, interpolation, json_):
        pt = create_pillow_transforms(json_)
        assert isinstance(pt, PillowPadToSize)
        assert pt.size == size
        assert pt.background_color == background_color
        assert pt.interpolation == interpolation

    @pytest.mark.parametrize(*tmatrix({
        'filename': [
            'nian_640_L.png',
            'nian_640_LA.png',
            'nian_640_RGB.png',
            'nian_640_RGBA.png',
            'png_640_m90.png',
            'png_640.png',
            'dori_640.png',
        ],
        'size': [
            (384, 512),
            (512, 384),
            (512, 512),
            (768, 512),
            (512, 768),
        ],
        'background_color': [
            'white',
            'black',
            'gray',
            'red',
            'green',
            'blue',
        ]
    }))
    def test_pad_to_size_exec(self, filename, size, background_color, image_diff):
        src_image = load_image(get_testfile(filename), mode=None, force_background=None)
        actual_image = PillowPadToSize(size=size, background_color=background_color)(src_image)
        body, ext = os.path.splitext(filename)
        expected_image_file = get_testfile(f'{body}_{size[0]}x{size[1]}_{background_color}.png')
        expected_image = load_image(expected_image_file, mode=None, force_background=None)
        assert image_diff(
            grid_transparent(actual_image),
            grid_transparent(expected_image),
            throw_exception=False
        ) < 1e-2

    @pytest.mark.parametrize(*tmatrix({
        'filename': [
            'nian_640_L.png',
            'nian_640_LA.png',
            'nian_640_RGB.png',
            'nian_640_RGBA.png',
            'png_640_m90.png',
            'png_640.png',
            'dori_640.png',
        ],
        'size': [384, 512, 768],
        'background_color': [
            'white',
            'black',
            'gray',
            'red',
            'green',
            'blue',
        ]
    }))
    def test_pad_to_size_int_size_exec(self, filename, size, background_color, image_diff):
        src_image = load_image(get_testfile(filename), mode=None, force_background=None)
        actual_image = PillowPadToSize(size=size, background_color=background_color)(src_image)
        body, ext = os.path.splitext(filename)
        expected_image_file = get_testfile(f'{body}_s{size}_{background_color}.png')
        expected_image = load_image(expected_image_file, mode=None, force_background=None)
        assert image_diff(
            grid_transparent(actual_image),
            grid_transparent(expected_image),
            throw_exception=False
        ) < 1e-2
