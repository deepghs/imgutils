import pytest

from imgutils.data import grid_transparent
from imgutils.metrics import psnr
from imgutils.restore import restore_with_scunet
from imgutils.restore.scunet import _open_scunet_model


@pytest.fixture(autouse=True, scope='module')
def _clear_cache():
    try:
        yield
    finally:
        _open_scunet_model.cache_clear()


@pytest.mark.unittest
class TestRestoreSCUNet:
    def test_restore_with_scunet_original(self, sample_image, clear_image):
        assert psnr(restore_with_scunet(sample_image), clear_image) >= 34.5

    def test_restore_with_scunet_q45(self, q45_image, clear_image):
        assert psnr(restore_with_scunet(q45_image), clear_image) >= 34.5

    def test_restore_with_scunet_gnoise(self, gaussian_noise_image, clear_image):
        assert psnr(restore_with_scunet(gaussian_noise_image), clear_image) >= 33

    def test_restore_with_scunet_rgba(self, rgba_image):
        assert rgba_image.mode == 'RGBA'
        restored_image = restore_with_scunet(rgba_image)
        assert restored_image.mode == 'RGBA'
        assert psnr(grid_transparent(restored_image), grid_transparent(rgba_image), ) >= 35
