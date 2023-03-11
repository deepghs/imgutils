import pytest
from PIL import Image

from imgutils.metrics import psnr
from test.testings import get_testfile


@pytest.mark.unittest
class TestMetricsPsnr:
    @pytest.mark.parametrize(['f1', 'f2', 'val'], [
        ('6124220.jpg', '6125785.png', 6.202801782403457),
        ('6125785.jpg', '6125785.png', 44.30738206970819),
        ('6125901.jpg', '6125785.png', 7.742781699128565),
    ])
    def test_psnr(self, f1, f2, val):
        img1 = Image.open(get_testfile(f1)).resize((512, 768))
        img2 = Image.open(get_testfile(f2)).resize((512, 768))
        assert psnr(img1, img2) == pytest.approx(val)
