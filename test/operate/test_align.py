import pytest
from PIL import Image

from imgutils.operate import align_maxsize
from test.testings import get_testfile


@pytest.fixture()
def genshin_post() -> Image.Image:
    return Image.open(get_testfile('genshin_post.jpg'))


@pytest.mark.unittest
class TestOperateAlign:
    def test_align_maxsize(self, genshin_post, image_diff):
        assert genshin_post.size == (1280, 720)
        _origin_mode = genshin_post.mode

        new_img1 = align_maxsize(genshin_post, 800)
        assert new_img1.size == (800, 450)
        assert new_img1.mode == _origin_mode
        assert image_diff(
            genshin_post.resize((800, 450)).convert('RGB'),
            new_img1.convert('RGB'),
            throw_exception=False
        ) < 1e-2

        new_img2 = align_maxsize(genshin_post, 2000)
        assert new_img2.size == (2000, 1125)
        assert new_img2.mode == _origin_mode
        assert image_diff(
            genshin_post.resize((2000, 1125)).convert('RGB'),
            new_img2.convert('RGB'),
            throw_exception=False
        ) < 1e-2
