from unittest import skipUnless

import pytest

from imgutils.preprocess.torchvision import _get_interpolation_mode

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

    def test_get_interpolation_mode_invalid(self):
        with pytest.raises(TypeError):
            _ = _get_interpolation_mode(None)
        with pytest.raises(TypeError):
            _ = _get_interpolation_mode([])
