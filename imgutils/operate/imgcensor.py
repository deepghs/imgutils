"""
Overview:
    Censors a specified area of an image using a custom image or emoji.
"""
import math
import os.path
from typing import Literal, Tuple, Optional

import numpy as np
from PIL import Image
from emoji import emojize
from hbutils.system import TemporaryDirectory
from scipy.ndimage import center_of_mass

from .align import align_maxsize
from .censor_ import BaseCensor, register_censor_method
from .squeeze import squeeze_with_transparency, _get_mask_of_transparency
from ..data import MultiImagesTyping, load_images
from ..utils import ts_lru_cache


def _image_rotate_and_sq(image: Image.Image, degrees: float):
    return squeeze_with_transparency(image.rotate(degrees, expand=True))


class SingleImage:
    """
    A class that attempts to find a solution to completely cover a given area of an image while minimizing the covered area.

    :param image: The input image.
    :type image: Image.Image

    :ivar image: The original image for censoring.
    :vartype image: Image.Image

    :ivar mask: The mask of the image. True means this pixel is not transparent and able to cover some area.
    :vartype mask: np.ndarray

    :ivar prefix: The prefix sum of the mask.
    :vartype prefix: np.ndarray

    :ivar cx: The X-coordinate of the mass center of this image. The position of the occlusion should be as close as
               possible to the mass center of the image.
    :vartype cx: float

    :ivar cy: The Y-coordinate of the mass center of this image. The position of the occlusion should be as close as
               possible to the mass center of the image.
    :vartype cy: float

    :ivar width: The width of the image.
    :vartype width: int

    :ivar height: The height of the image.
    :vartype height: int
    """

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
        self.cx, self.cy = center_of_mass(mask)

    @property
    def width(self):
        """
        The width of the image.

        :return: The width of the image.
        :rtype: int
        """
        return self.mask.shape[0]

    @property
    def height(self):
        """
        The height of the image.

        :return: The height of the image.
        :rtype: int
        """
        return self.mask.shape[1]

    def _find_for_fixed_area(self, width: int, height: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Finds a solution for a fixed area (width and height) that can completely cover the given area.

        :param width: The width of the fixed area.
        :type width: int

        :param height: The height of the fixed area.
        :type height: int

        :return: The coordinates of the found solution (top-left corner) or None if no solution is found.
        :rtype: Tuple[Optional[int], Optional[int]]
        """
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
        """
        Finds a solution to completely cover a given area with a rectangle while minimizing the covered area.

        :param width: The width of the target area to cover.
        :type width: int

        :param height: The height of the target area to cover.
        :type height: int

        :return: The coordinates (x, y) of the found solution (top-left corner), the scaling factor applied to the
                 width and height, and the ratio of the covered area to the total mask area.
        :rtype: Tuple[float, float, float, float]
        """
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
    """
    A class that performs censoring on a given area using images by finding the best solution based on rotation.

    :param images: The input images for censoring.
    :type images: MultiImagesTyping

    :param rotate: The range of rotation angles in degrees (start, end) to consider.
    :type rotate: Tuple[int, int]

    :param step: The step size between rotation angles.
    :type step: int
    """

    def __init__(self, images: MultiImagesTyping, rotate: Tuple[int, int] = (-30, 30), step: int = 10):
        origin_images = load_images(images, mode='RGBA', force_background=None)
        degrees = sorted(list(range(rotate[0], rotate[1], step)), key=lambda x: (abs(x), x))
        self.images = [
            SingleImage(_image_rotate_and_sq(img, d))
            for d in degrees for img in origin_images
        ]

    def _find_censor(self, area: Tuple[int, int, int, int], ratio_threshold: float = 0.5):
        """
        Finds the best censoring solution for the given area using the available images.

        :param area: The coordinates of the target area to censor (x0, y0, x1, y1).
        :type area: Tuple[int, int, int, int]

        :param ratio_threshold: The minimum ratio of the covered area to the total mask area required for a solution
                                to be considered valid.
        :type ratio_threshold: float

        :return: The ratio of the covered area to the total mask area, the index of the selected image, the scaling
                 factor applied to the image, the X-coordinate of the top-left corner of the censoring area, and the
                 Y-coordinate of the top-left corner of the censoring area.
        :rtype: Tuple[float, int, float, float, float]
        """
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
        """
        Censors the specified area of the input image using the available images.

        :param image: The input image to be censored.
        :type image: Image.Image

        :param area: The coordinates of the target area to censor (x0, y0, x1, y1).
        :type area: Tuple[int, int, int, int]

        :param ratio_threshold: The minimum ratio of the covered area to the total mask area required for a solution
                                to be considered valid.
        :type ratio_threshold: float

        :param kwargs: Additional keyword arguments to be passed.

        :return: The censored image.
        :rtype: Image.Image

        Examples::
            >>> from PIL import Image
            >>> from imgutils.operate import censor_areas
            >>>
            >>> origin = Image.open('genshin_post.jpg')
            >>> areas = [  # areas to censor
            >>>     (967, 143, 1084, 261),
            >>>     (246, 208, 331, 287),
            >>>     (662, 466, 705, 514),
            >>>     (479, 283, 523, 326)
            >>> ]
            >>>
            >>> # register the star image
            >>> register_censor_method('star', ImageBasedCensor, images=['star.png'])
            >>>
            >>> # default
            >>> censored = censor_areas(image, 'star', areas)

            .. image:: censor_image.plot.py.svg
                :align: center

            .. note::
                It is important to note that when using :class:`ImageBasedCensor` to censor an image,
                you need to manually register the image used for censoring
                using the :func:`register_censor_method` function.

        """
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


@ts_lru_cache()
def _get_emoji_img(emoji: str, style: str = 'twitter') -> Image.Image:
    from pilmoji.source import EmojiCDNSource

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


@ts_lru_cache()
def _get_native_emoji_censor(emoji: str = ':smiling_face_with_heart-eyes:', style: _EmojiStyleTyping = 'twitter',
                             rotate: Tuple[int, int] = (-30, 30), step: int = 10):
    return _NativeEmojiBasedCensor(emoji, style, rotate, step)


class EmojiBasedCensor(BaseCensor):
    """
    Performs censoring on a given area of an image using emoji images.

    :param rotate: The range of rotation angles in degrees (start, end) to consider.
    :type rotate: Tuple[int, int]

    :param step: The step size between rotation angles.
    :type step: int
    """

    def __init__(self, rotate: Tuple[int, int] = (-30, 30), step: int = 10):
        self.rotate = rotate
        self.step = step

    def censor_area(self, image: Image.Image, area: Tuple[int, int, int, int],
                    emoji: str = ':smiling_face_with_heart-eyes:', style: _EmojiStyleTyping = 'twitter',
                    ratio_threshold: float = 0.5, **kwargs) -> Image.Image:
        """
        Censors the specified area of the input image using emoji expressions.

        :param image: The input image to be censored.
        :type image: Image.Image

        :param area: The coordinates of the target area to censor (x0, y0, x1, y1).
        :type area: Tuple[int, int, int, int]

        :param emoji: The emoji expression to use for censoring.
                      Emoji code in `emoji <https://github.com/carpedm20/emoji>`_ is supported.
                      (default: ``:smiling_face_with_heart-eyes:``, which equals to 😍)
        :type emoji: str

        :param style: The style of the emoji expression. (default: ``twitter``)
        :type style: _EmojiStyleTyping

        :param ratio_threshold: The minimum ratio of the covered area to the total mask area required for a solution
                                to be considered valid.
        :type ratio_threshold: float

        :param kwargs: Additional keyword arguments to be passed.

        :return: The censored image.
        :rtype: Image.Image

        Examples::
            >>> from PIL import Image
            >>> from imgutils.operate import censor_areas
            >>>
            >>> origin = Image.open('genshin_post.jpg')
            >>> areas = [  # areas to censor
            >>>     (967, 143, 1084, 261),
            >>>     (246, 208, 331, 287),
            >>>     (662, 466, 705, 514),
            >>>     (479, 283, 523, 326)
            >>> ]
            >>>
            >>> # default
            >>> emoji_default = censor_areas(image, 'emoji', areas)
            >>>
            >>> # cat_face (use emoji code)
            >>> emoji_green = censor_areas(image, 'emoji', areas, emoji=':cat_face:')
            >>>
            >>> # grinning_face_with_sweat (use emoji)
            >>> emoji_liuhanhuangdou = censor_areas(image, 'emoji', areas, emoji='😅')

            This is the result:

            .. image:: censor_emoji.plot.py.svg
                :align: center
        """
        return _get_native_emoji_censor(emoji, style, self.rotate, self.step) \
            .censor_area(image, area, ratio_threshold, **kwargs)


register_censor_method('emoji', EmojiBasedCensor)
