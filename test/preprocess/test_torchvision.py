from unittest import skipUnless

import pytest
from PIL import Image
from hbutils.testing import tmatrix

from imgutils.preprocess.torchvision import _get_interpolation_mode, create_torchvision_transforms
from test.testings import get_testfile

try:
    import torchvision
except (ImportError, ModuleNotFoundError):
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


@pytest.mark.unittest
class TestPreprocessPillow:
    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_get_interpolation_mode_raw(self):
        from torchvision.transforms import InterpolationMode
        assert _get_interpolation_mode(InterpolationMode.NEAREST) == InterpolationMode.NEAREST
        assert _get_interpolation_mode(InterpolationMode.LANCZOS) == InterpolationMode.LANCZOS
        assert _get_interpolation_mode(InterpolationMode.BILINEAR) == InterpolationMode.BILINEAR
        assert _get_interpolation_mode(InterpolationMode.BICUBIC) == InterpolationMode.BICUBIC
        assert _get_interpolation_mode(InterpolationMode.BOX) == InterpolationMode.BOX
        assert _get_interpolation_mode(InterpolationMode.HAMMING) == InterpolationMode.HAMMING

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_get_interpolation_mode_int(self):
        from torchvision.transforms import InterpolationMode
        assert _get_interpolation_mode(0) == InterpolationMode.NEAREST
        assert _get_interpolation_mode(1) == InterpolationMode.LANCZOS
        assert _get_interpolation_mode(2) == InterpolationMode.BILINEAR
        assert _get_interpolation_mode(3) == InterpolationMode.BICUBIC
        assert _get_interpolation_mode(4) == InterpolationMode.BOX
        assert _get_interpolation_mode(5) == InterpolationMode.HAMMING
        with pytest.raises(ValueError):
            _get_interpolation_mode(-1)
        with pytest.raises(ValueError):
            _get_interpolation_mode(100)

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_get_interpolation_mode_str(self):
        from torchvision.transforms import InterpolationMode
        assert _get_interpolation_mode('nearest') == InterpolationMode.NEAREST
        assert _get_interpolation_mode('NEAREST') == InterpolationMode.NEAREST
        assert _get_interpolation_mode('bilinear') == InterpolationMode.BILINEAR
        assert _get_interpolation_mode('bicubic') == InterpolationMode.BICUBIC
        assert _get_interpolation_mode('box') == InterpolationMode.BOX
        assert _get_interpolation_mode('hamming') == InterpolationMode.HAMMING
        assert _get_interpolation_mode('lanczos') == InterpolationMode.LANCZOS
        with pytest.raises(ValueError):
            _ = _get_interpolation_mode('xxx')
        with pytest.raises(ValueError):
            _ = _get_interpolation_mode('')

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_get_interpolation_mode_invalid(self):
        with pytest.raises(TypeError):
            _ = _get_interpolation_mode(None)
        with pytest.raises(TypeError):
            _ = _get_interpolation_mode([])

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        ('mode', 'channels'): [
            ('I', 1), ('I;16', 1), ('F', 1), ('1', 1), ('L', 1), ('P', 1),
            ('LA', 2),
            ('RGB', 3), ('YCbCr', 3),
            ('RGBA', 4), ('CMYK', 4),
        ]
    }))
    def test_maybe_to_tensor(self, src_image, mode, channels):
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode

        ttrans = create_torchvision_transforms({'type': 'maybe_to_tensor'})
        result = ttrans(image)
        assert tuple(result.shape) == (channels, image.height, image.width)

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_maybe_to_tensor_repr(self):
        ttrans = create_torchvision_transforms({'type': 'maybe_to_tensor'})
        assert repr(ttrans) == 'MaybeToTensor()'

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_maybe_to_tensor_np(self):
        import torch
        input_ = torch.randn(3, 384, 384)
        ttrans = create_torchvision_transforms({'type': 'maybe_to_tensor'})
        assert torch.allclose(ttrans(input_), input_)

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        ('mode', 'channels'): [
            ('I', 1), ('I;16', 1), ('F', 1), ('1', 1), ('L', 1), ('P', 1),
            ('LA', 2),
            ('RGB', 3), ('YCbCr', 3),
            ('RGBA', 4), ('CMYK', 4),
        ]
    }))
    def test_to_tensor(self, src_image, mode, channels):
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode

        ttrans = create_torchvision_transforms({'type': 'to_tensor'})
        result = ttrans(image)
        assert tuple(result.shape) == (channels, image.height, image.width)

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_create_transform_invalid(self):
        with pytest.raises(TypeError):
            _ = create_torchvision_transforms(None)
        with pytest.raises(TypeError):
            _ = create_torchvision_transforms(1)
        with pytest.raises(TypeError):
            _ = create_torchvision_transforms('str')

    @skipUnless(not _TORCHVISION_AVAILABLE, 'Non-torchvision required.')
    def test_create_transform_non_torchvision(self):
        with pytest.raises(EnvironmentError):
            _ = create_torchvision_transforms([])
