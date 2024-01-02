import pytest

from imgutils.metrics import psnr
from imgutils.restore import restore_with_nafnet
from imgutils.restore.nafnet import _open_nafnet_model


@pytest.fixture(autouse=True, scope='module')
def _clear_cache():
    try:
        yield
    finally:
        _open_nafnet_model.cache_clear()


@pytest.mark.unittest
class TestRestoreNafNet:
    def test_restore_with_nafnet_original(self, sample_image, clear_image):
        assert psnr(restore_with_nafnet(sample_image), clear_image) >= 40.0

    def test_restore_with_nafnet_q45(self, q45_image, clear_image):
        assert psnr(restore_with_nafnet(q45_image), clear_image) >= 40.0
