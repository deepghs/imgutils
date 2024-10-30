import pytest
from PIL import Image

from imgutils.data import load_image, has_alpha_channel, add_background_for_rgba
from test.testings import get_testfile

_FILENAME = get_testfile('6125785.png')
_IMAGE = Image.open(_FILENAME)


@pytest.mark.unittest
class TestDataImage:
    @pytest.mark.parametrize(['image_', 'result'], [
        (_FILENAME, _IMAGE),
        (_IMAGE, _IMAGE),
        (None, TypeError),
    ])
    def test_load_image(self, image_, result, image_diff):
        if isinstance(result, type) and issubclass(result, BaseException):
            with pytest.raises(result):
                _ = load_image(image_)
        elif isinstance(image_, Image.Image):
            assert load_image(image_, force_background=None) is image_
        else:
            assert image_diff(load_image(image_), result, throw_exception=False) < 1e-2

    @pytest.mark.parametrize(['color'], [
        ('white',),
        ('green',),
        ('red',),
        ('blue',),
        ('black',),
    ])
    def test_load_image_bg_rgba(self, image_diff, color):
        image = load_image(get_testfile('nian.png'), force_background=color, mode='RGB')
        expected = Image.open(get_testfile(f'nian_bg_{color}.png'))
        assert image_diff(image, expected, throw_exception=False) < 1e-2

    @pytest.mark.parametrize(['color'], [
        ('white',),
        ('green',),
        ('red',),
        ('blue',),
        ('black',),
    ])
    def test_add_background_for_rgba_rgba(self, image_diff, color):
        image = add_background_for_rgba(get_testfile('nian.png'), background=color)
        assert image.mode == 'RGB'
        expected = Image.open(get_testfile(f'nian_bg_{color}.png'))
        assert image_diff(image, expected, throw_exception=False) < 1e-2

    @pytest.mark.parametrize(['color'], [
        ('white',),
        ('green',),
        ('red',),
        ('blue',),
        ('black',),
    ])
    def test_load_image_bg_rgb(self, image_diff, color):
        image = load_image(get_testfile('mostima_post.jpg'), force_background=color, mode='RGB')
        expected = Image.open(get_testfile(f'mostima_post_bg_{color}.png'))
        assert image_diff(image, expected, throw_exception=False) < 1e-2

    @pytest.mark.parametrize(['color'], [
        ('white',),
        ('green',),
        ('red',),
        ('blue',),
        ('black',),
    ])
    def test_add_backround_for_rgba_rgb(self, image_diff, color):
        image = add_background_for_rgba(get_testfile('mostima_post.jpg'), background=color)
        assert image.mode == 'RGB'
        expected = Image.open(get_testfile(f'mostima_post_bg_{color}.png'))
        assert image_diff(image, expected, throw_exception=False) < 1e-2


@pytest.fixture
def rgba_image():
    img = Image.new('RGBA', (10, 10), (255, 0, 0, 128))
    return img


@pytest.fixture
def rgb_image():
    img = Image.new('RGB', (10, 10), (255, 0, 0))
    return img


@pytest.fixture
def la_image():
    img = Image.new('LA', (10, 10), (128, 128))
    return img


@pytest.fixture
def l_image():
    img = Image.new('L', (10, 10), 128)
    return img


@pytest.fixture
def p_image_with_transparency():
    width, height = 200, 200
    image = Image.new('P', (width, height))

    palette = []
    for i in range(256):
        palette.extend((i, i, i))  # 灰度调色板

    palette[:3] = (0, 0, 0)  # 黑色
    image.info['transparency'] = 0

    image.putpalette(palette)
    return image


@pytest.fixture
def p_image_without_transparency():
    img = Image.new('P', (10, 10))
    palette = [255, 0, 0, 255, 0, 0]  # No transparent color
    img.putpalette(palette)
    return img


@pytest.mark.unittest
class TestHasAlphaChannel:
    def test_rgba_image(self, rgba_image):
        assert has_alpha_channel(rgba_image)

    def test_rgb_image(self, rgb_image):
        assert not has_alpha_channel(rgb_image)

    def test_la_image(self, la_image):
        assert has_alpha_channel(la_image)

    def test_l_image(self, l_image):
        assert not has_alpha_channel(l_image)

    def test_p_image_with_transparency(self, p_image_with_transparency):
        assert has_alpha_channel(p_image_with_transparency)

    def test_p_image_without_transparency(self, p_image_without_transparency):
        assert not has_alpha_channel(p_image_without_transparency)

    def test_pa_image(self):
        pa_image = Image.new('PA', (10, 10))
        assert has_alpha_channel(pa_image)
