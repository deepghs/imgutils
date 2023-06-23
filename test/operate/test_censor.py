import re
from typing import List, Tuple
from unittest import skipUnless

import pytest
from PIL import Image
from emoji import emojize, demojize
from hbutils.testing import vpython, tmatrix

from imgutils.operate import censor_nsfw, register_censor_method, ImageBasedCensor, censor_areas
from test.testings import get_testfile


@pytest.fixture()
def nude_girl() -> Image.Image:
    return Image.open(get_testfile('nude_girl.png'))


@pytest.fixture()
def complex_sex() -> Image.Image:
    return Image.open(get_testfile('complex_sex.jpg'))


@pytest.fixture()
def genshin_post() -> Image.Image:
    return Image.open(get_testfile('genshin_post.jpg'))


@pytest.fixture()
def gp_areas() -> List[Tuple[int, int, int, int]]:
    return [
        (967, 143, 1084, 261),
        (246, 208, 331, 287),
        (662, 466, 705, 514),
        (479, 283, 523, 326)
    ]


@pytest.mark.unittest
class TestOperateCensor:
    @pytest.mark.parametrize(*tmatrix({
        'nipple_f': [True, False],
        'penis': [True, False],
        'pussy': [True, False],
        'method': ['color', 'pixelate', 'blur', *(('emoji',) if vpython >= '3.8' else ())],
    }, mode='matrix'))
    def test_censor_nsfw(self, complex_sex, nipple_f, penis, pussy, method, image_diff):
        dst_filename = f'complex_sex_{"nipple_f" if nipple_f else "o"}' \
                       f'_{"penis" if penis else "o"}_{"pussy" if pussy else "o"}_{method}.jpg'
        censored = censor_nsfw(complex_sex, method, nipple_f, penis, pussy)
        assert image_diff(
            censored.convert('RGB'),
            Image.open(get_testfile(dst_filename)).convert('RGB'),
            throw_exception=False,
        ) < 1e-2

    @pytest.mark.parametrize(['color'], [('red',), ('black',), ('green',), ('blue',)])
    def test_censor_color(self, genshin_post, gp_areas, color, image_diff):
        assert image_diff(
            censor_areas(genshin_post, 'color', gp_areas, color=color).convert('RGB'),
            Image.open(get_testfile(f'genshin_post_color_{color}.jpg')).convert('RGB'),
            throw_exception=False
        ) < 1e-2

    @pytest.mark.parametrize(['radius'], [(4,), (8,), (12,)])
    def test_censor_blur(self, genshin_post, gp_areas, radius, image_diff):
        assert image_diff(
            censor_areas(genshin_post, 'blur', gp_areas, radius=radius).convert('RGB'),
            Image.open(get_testfile(f'genshin_post_blur_{radius}.jpg')).convert('RGB'),
            throw_exception=False
        ) < 1e-2

    @pytest.mark.parametrize(['radius'], [(4,), (8,), (12,)])
    def test_censor_pixelate(self, genshin_post, gp_areas, radius, image_diff):
        assert image_diff(
            censor_areas(genshin_post, 'pixelate', gp_areas, radius=radius).convert('RGB'),
            Image.open(get_testfile(f'genshin_post_pixelate_{radius}.jpg')).convert('RGB'),
            throw_exception=False
        ) < 1e-2

    @skipUnless(vpython >= '3.8', 'Python3.8+ required')
    @pytest.mark.parametrize(['emoji'], [
        (':smiling_face_with_heart-eyes:',),
        (':cat_face:',),
        ('😅',),
    ])
    def test_censor_emoji(self, genshin_post, gp_areas, emoji, image_diff):
        emoji_name = re.sub(r'[\W_]+', '_', demojize(emojize(emoji))).strip('_')
        assert image_diff(
            censor_areas(genshin_post, 'emoji', gp_areas, emoji=emoji).convert('RGB'),
            Image.open(get_testfile(f'genshin_post_emoji_{emoji_name}.jpg')).convert('RGB'),
            throw_exception=False
        ) < 1e-2

    @skipUnless(vpython < '3.8', 'Python3.8+ required')
    @pytest.mark.parametrize(['emoji'], [
        (':smiling_face_with_heart-eyes:',),
        (':cat_face:',),
        ('😅',),
    ])
    def test_censor_emoji_py37(self, genshin_post, gp_areas, emoji, image_diff):
        emoji_name = re.sub(r'[\W_]+', '_', demojize(emojize(emoji))).strip('_')
        try:
            with pytest.warns(Warning):
                assert image_diff(
                    censor_areas(genshin_post, 'emoji', gp_areas, emoji=emoji).convert('RGB'),
                    Image.open(get_testfile(f'genshin_post_emoji_py37_{emoji_name}.jpg')).convert('RGB'),
                    throw_exception=False
                ) < 1e-2
        finally:
            from imgutils.operate.imgcensor import _py37_fallback
            _py37_fallback.cache_clear()

    def test_unknown_censor_method(self, complex_sex):
        with pytest.raises(KeyError):
            _ = censor_nsfw(complex_sex, method='unknown_method', nipple_f=True)

    def test_register_censor_method(self, complex_sex, image_diff):
        register_censor_method('star', ImageBasedCensor, images=[get_testfile('star.png')])

        assert image_diff(
            censor_nsfw(complex_sex, method='star', nipple_f=True).convert('RGB'),
            Image.open(get_testfile('complex_sex_star_censored.jpg')).convert('RGB'),
            throw_exception=False,
        ) < 1e-2

    def test_register_censor_method_failed(self):
        with pytest.raises(KeyError):
            _ = register_censor_method('emoji', ImageBasedCensor, images=[get_testfile('star.png')])
