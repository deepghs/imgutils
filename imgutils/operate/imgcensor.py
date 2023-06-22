import math
import os.path
import warnings
from functools import lru_cache
from typing import Tuple, Optional

import numpy as np
from PIL import Image
from emoji import emojize
from hbutils.system import TemporaryDirectory
from hbutils.testing import vpython
from pilmoji.source import EmojiCDNSource
from scipy import ndimage

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal

from .align import align_maxsize
from .censor_ import BaseCensor, register_censor_method
from .squeeze import squeeze_with_transparency, _get_mask_of_transparency
from ..data import MultiImagesTyping, load_images


def _image_rotate_and_sq(image: Image.Image, degrees: float):
    return squeeze_with_transparency(image.rotate(degrees, expand=True))


class SingleImage:
    def __init__(self, image: Image.Image):
        mask = _get_mask_of_transparency(align_maxsize(image, 300))
        mask = mask.transpose((1, 0)).astype(np.uint64)
        prefix = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=mask.dtype)
        prefix[1:, 1:] = np.cumsum(np.cumsum(mask, axis=1), axis=0)

        # original image for censoring
        # do not use self.image inside this class,
        # because its size is not assumed to be the same as self.mask.shape
        self.image = image

        # mask of the image (True means this pixel is not transparent
        # and able to cover some area)
        self.mask = mask

        # prefix sum of the mask
        self.prefix = prefix

        # mass center of this image, the position of the occlusion
        # should be as close as possible to the mass center of the image
        self.cx, self.cy = ndimage.measurements.center_of_mass(mask)

    @property
    def width(self):
        return self.mask.shape[0]

    @property
    def height(self):
        return self.mask.shape[1]

    def _find_for_fixed_area(self, width: int, height: int) -> Tuple[Optional[int], Optional[int]]:
        if width > self.mask.shape[0] or height > self.mask.shape[1]:
            return None, None

        delta = self.prefix[width:, height:] - self.prefix[:-width, height:] - \
                self.prefix[width:, :-height] + self.prefix[:-width, :-height]
        assert delta.shape == (self.mask.shape[0] - width + 1, self.mask.shape[1] - height + 1)

        xs, ys = np.where(delta == width * height)
        if len(xs) == 0:  # not found
            return None, None

        centers = np.stack([xs + width // 2, ys + height // 2]).transpose((1, 0))
        best_cid = np.argmin(((centers - np.array([[self.cx, self.cy]])) ** 2).sum(axis=1))
        fx, fy = centers[best_cid]
        return int(fx), int(fy)

    def find_for_area(self, width: int, height: int) -> Tuple[float, float, float, float]:
        l, r = 0.0, 1.0 / max(*self.mask.shape)
        while True:
            new_width, new_height = int(math.ceil(width * r)), int(math.ceil(height * r))
            fx, fy = self._find_for_fixed_area(new_width, new_height)
            if fx is not None:
                l, r = r, r * 2
            else:
                break

        eps = 1e-6
        r_fx, r_fy = None, None
        while l + eps < r:
            m = (l + r) / 2
            new_width, new_height = int(math.ceil(width * m)), int(math.ceil(height * m))
            fx, fy = self._find_for_fixed_area(new_width, new_height)
            if fx is not None:
                r_fx, r_fy = fx - new_width / 2, fy - new_height / 2
                l = m
            else:
                r = m

        ratio = (width * l) * (height * l) / (self.mask.sum())
        return r_fx, r_fy, l, ratio


class ImageBasedCensor(BaseCensor):
    def __init__(self, images: MultiImagesTyping, rotate: Tuple[int, int] = (-30, 30), step: int = 10):
        origin_images = load_images(images, mode='RGBA', force_background=None)
        degrees = sorted(list(range(rotate[0], rotate[1], step)), key=lambda x: (abs(x), x))
        self.images = [
            SingleImage(_image_rotate_and_sq(img, d))
            for d in degrees for img in origin_images
        ]

    def _find_censor(self, area: Tuple[int, int, int, int], ratio_threshold: float = 0.5):
        x0, y0, x1, y1 = area
        width, height = x1 - x0, y1 - y0

        results = []
        for i, m_image in enumerate(self.images):
            r_fx, r_fy, scale, ratio = m_image.find_for_area(width, height)
            results.append((ratio, i, scale, r_fx, r_fy))
            if ratio > ratio_threshold:
                return ratio, i, scale, r_fx, r_fy

        ratio, idx, scale, r_fx, r_fy = sorted(results, key=lambda x: -x[0])[0]
        return ratio, idx, scale, r_fx, r_fy

    def censor_area(self, image: Image.Image, area: Tuple[int, int, int, int], ratio_threshold: float = 0.5,
                    **kwargs) -> Image.Image:
        x0, y0, x1, y1 = area
        ratio, idx, scale, r_fx, r_fy = self._find_censor(area, ratio_threshold)
        fm_image = self.images[idx]
        censor_image = fm_image.image.copy()
        censor_image = censor_image.resize((  # do not use censor_image.size here
            int(math.ceil(fm_image.width / scale)),
            int(math.ceil(fm_image.height / scale)),
        ))
        cx0, cy0 = int(x0 - r_fx / scale), int(y0 - r_fy / scale)

        mode = image.mode
        image = image.copy().convert('RGBA')
        image.paste(censor_image, (cx0, cy0, cx0 + censor_image.width, cy0 + censor_image.height), mask=censor_image)
        return image.convert(mode)


def _get_file_in_censor_assets(file):
    return os.path.normpath(os.path.join(__file__, '..', file))


register_censor_method(
    'heart', ImageBasedCensor,
    images=[_get_file_in_censor_assets('heart_censor.png')]
)
register_censor_method(
    'smile', ImageBasedCensor,
    images=[_get_file_in_censor_assets('smile_censor.png')]
)


@lru_cache()
def _get_emoji_img(emoji: str, style: str = 'twitter') -> Image.Image:
    class _CustomSource(EmojiCDNSource):
        STYLE = style

    with TemporaryDirectory() as td:
        imgfile = os.path.join(td, 'emoji.png')
        with open(imgfile, 'wb') as f:
            f.write(_CustomSource().get_emoji(emojize(emoji)).read())

        img = Image.open(imgfile)
        img.load()
        return img


_EmojiStyleTyping = Literal[
    'twitter', 'apple', 'google', 'microsoft', 'samsung', 'whatsapp', 'facebook', 'messenger',
    'joypixels', 'openmoji', 'emojidex', 'mozilla'
]


class _NativeEmojiBasedCensor(ImageBasedCensor):
    def __init__(self, emoji: str = ':smiling_face_with_heart-eyes:', style: _EmojiStyleTyping = 'twitter',
                 rotate: Tuple[int, int] = (-30, 30), step: int = 10):
        ImageBasedCensor.__init__(self, [_get_emoji_img(emoji, style)], rotate, step)


@lru_cache()
def _get_native_emoji_censor(emoji: str = ':smiling_face_with_heart-eyes:', style: _EmojiStyleTyping = 'twitter',
                             rotate: Tuple[int, int] = (-30, 30), step: int = 10):
    return _NativeEmojiBasedCensor(emoji, style, rotate, step)


class EmojiBasedCensor(BaseCensor):
    def __init__(self, rotate: Tuple[int, int] = (-30, 30), step: int = 10):
        self.rotate = rotate
        self.step = step

    def censor_area(self, image: Image.Image, area: Tuple[int, int, int, int],
                    emoji: str = ':smiling_face_with_heart-eyes:', style: _EmojiStyleTyping = 'twitter',
                    ratio_threshold: float = 0.5, **kwargs) -> Image.Image:
        return _get_native_emoji_censor(emoji, style, self.rotate, self.step) \
            .censor_area(image, area, ratio_threshold, **kwargs)


if vpython >= '3.8':
    register_censor_method('emoji', EmojiBasedCensor)

else:
    @lru_cache()
    def _py37_fallback():
        warnings.warn('Due to compatibility issues with the pilmoji library, '
                      'the emoji censor method is not supported in Python 3.7. '
                      'A pre-defined single emoji image will be used for rendering, '
                      'and the emoji and style parameters will be ignored.')


    class _Python37FallbackCensor(ImageBasedCensor):
        def __init__(self):
            ImageBasedCensor.__init__(self, [_get_file_in_censor_assets('emoji_censor.png')])

        def censor_area(self, image: Image.Image, area: Tuple[int, int, int, int], ratio_threshold: float = 0.5,
                        **kwargs) -> Image.Image:
            _py37_fallback()
            return ImageBasedCensor.censor_area(self, image, area, ratio_threshold, **kwargs)


    register_censor_method('emoji', _Python37FallbackCensor)
