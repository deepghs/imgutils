import pytest

from imgutils.metrics import laplacian_score
from test.testings import get_testfile


@pytest.mark.unittest
class TestClusteringLaplacian:
    @pytest.mark.parametrize(['image_file', 'expected_score'], [
        ('text_blur.png', 2276.66629157129),
        ('hutao.jpg', 156.68285005210006),
        ('real2.png', 15.908745781486806),
        ('mmd.png', 1072.8372572065527),
    ])
    def test_laplacian_score(self, image_file, expected_score):
        assert laplacian_score(get_testfile('laplacian', image_file)) == pytest.approx(expected_score)
