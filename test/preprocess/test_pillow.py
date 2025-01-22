from unittest import skipUnless

import numpy as np
import pytest
from PIL import Image
from hbutils.testing import tmatrix

from imgutils.preprocess.pillow import PillowResize, _get_pillow_resample, PillowCenterCrop, PillowToTensor
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

    def test_to_tensor_invalid(self):
        ptotensor = PillowToTensor()
        with pytest.raises(TypeError):
            _ = ptotensor(np.random.randn(3, 384, 384))

    def test_to_tensor_repr(self):
        return repr(PillowToTensor()) == 'PillowToTensor()'
