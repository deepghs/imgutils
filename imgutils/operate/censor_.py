from functools import lru_cache
from typing import Tuple, Type, List

from PIL import Image, ImageFilter

from ..data import ImageTyping, load_image
from ..detect import detect_censors


class BaseCensor:
    """
    The Censor base class serves as the foundation for creating custom censor methods by inheriting
    from this class and registering them using the :func:`register_censor_method` function.
    """

    def censor_area(self, image: Image.Image, area: Tuple[int, int, int, int], **kwargs) -> Image.Image:
        """
        Applies censoring to a specific area within the image.

        :param image: An instance of PIL Image representing the input image.
        :type image: Image.Image

        :param area: A tuple representing the rectangular area to be censored
            in the format ``(left, upper, right, lower)``.
        :type area: Tuple[int, int, int, int]

        :param kwargs: Additional keyword arguments for customization.

        :return: An instance of PIL Image with the censored area.
        :rtype: Image.Image
        """
        raise NotImplementedError  # pragma: no cover


class PixelateCensor(BaseCensor):
    """
    A class that performs pixelization censoring on a specific area of an image.

    Inherits from :class:`BaseCensor`.
    """

    def censor_area(self, image: Image.Image, area: Tuple[int, int, int, int], radius: int = 4,
                    **kwargs) -> Image.Image:
        """
        Applies pixelization censoring to a specific area within the image.

        :param image: An instance of PIL Image representing the input image.
        :type image: Image.Image

        :param area: A tuple representing the rectangular area to be censored
            in the format ``(left, upper, right, lower)``.
        :type area: Tuple[int, int, int, int]

        :param radius: The radius of the pixelation effect. Default is ``4``.
        :type radius: int

        :param kwargs: Additional keyword arguments for customization.

        :return: An instance of PIL Image with the pixelated area.
        :rtype: Image.Image
        """
        image = image.copy()
        x0, y0, x1, y1 = area
        width, height = x1 - x0, y1 - y0
        censor_area = image.crop((x0, y0, x1, y1))
        censor_area = censor_area.resize((width // radius, height // radius)).resize((width, height), Image.NEAREST)
        image.paste(censor_area, (x0, y0, x1, y1))
        return image


class BlurCensor(BaseCensor):
    """
    A class that performs blurring censoring on a specific area of an image.

    Inherits from :class:`BaseCensor`.
    """

    def censor_area(self, image: Image.Image, area: Tuple[int, int, int, int], radius: int = 4,
                    **kwargs) -> Image.Image:
        """
        Applies blurring censoring to a specific area within the image.

        :param image: An instance of PIL Image representing the input image.
        :type image: Image.Image

        :param area: A tuple representing the rectangular area to be censored
            in the format ``(left, upper, right, lower)``.
        :type area: Tuple[int, int, int, int]

        :param radius: The radius of the blurring effect. Default is ``4``.
        :type radius: int

        :param kwargs: Additional keyword arguments for customization.

        :return: An instance of PIL Image with the blurred area.
        :rtype: Image.Image
        """
        image = image.copy()
        x0, y0, x1, y1 = area
        censor_area = image.crop((x0, y0, x1, y1))
        censor_area = censor_area.filter(ImageFilter.GaussianBlur(radius))
        image.paste(censor_area, (x0, y0, x1, y1))
        return image


class ColorCensor(BaseCensor):
    """
    A class that performs color censoring by filling a specific area of an image with a solid color.

    Inherits from :class:`BaseCensor`.
    """

    def censor_area(self, image: Image.Image, area: Tuple[int, int, int, int], color: str = 'black',
                    **kwargs) -> Image.Image:
        """
        Fills a specific area within the image with a solid color for censoring.

        :param image: An instance of PIL Image representing the input image.
        :type image: Image.Image

        :param area: A tuple representing the rectangular area to be censored
            in the format ``(left, upper, right, lower)``.
        :type area: Tuple[int, int, int, int]

        :param color: The color used to fill the censor area. Default is ``black``.
            Can be any valid color name or RGB value.
        :type color: str

        :param kwargs: Additional keyword arguments for customization.

        :return: An instance of PIL Image with the censored area filled with the specified color.
        :rtype: Image.Image
        """
        image = image.copy()
        x0, y0, x1, y1 = area
        censor_area = image.crop((x0, y0, x1, y1))
        # noinspection PyTypeChecker
        censor_area = Image.new(image.mode, censor_area.size, color=color)
        image.paste(censor_area, (x0, y0, x1, y1))
        return image


_KNOWN_CENSORS = {}


def register_censor_method(name: str, cls: Type[BaseCensor], *args, **kwargs):
    """
    Overview:
        Registers a censor method for subsequent censoring tasks.

    :param name: The name of the censor method.
    :type name: str

    :param cls: The class representing the censor method. It should be a subclass of BaseCensor.
    :type cls: Type[BaseCensor]

    :param args: Positional arguments to be passed when initializing the censor method.
    :param kwargs: Keyword arguments to be passed when initializing the censor method.

    :raises KeyError: If the censor method name already exists.

    """
    if name in _KNOWN_CENSORS:
        raise KeyError(f'Censor method {name!r} already exist, please use another name.')
    _KNOWN_CENSORS[name] = (cls, args, kwargs)


register_censor_method('pixelate', PixelateCensor)
register_censor_method('blur', BlurCensor)
register_censor_method('color', ColorCensor)


@lru_cache()
def _get_censor_instance(name: str) -> BaseCensor:
    if name in _KNOWN_CENSORS:
        cls, args, kwargs = _KNOWN_CENSORS[name]
        return cls(*args, **kwargs)
    else:
        raise KeyError(f'Censor method {name!r} not found.')


def censor_areas(image: ImageTyping, method: str,
                 areas: List[Tuple[float, float, float, float]], **kwargs) -> Image.Image:
    """
    Applies censoring to specific areas of an image using the registered censor method.

    :param image: The input image to be censored.
    :type image: ImageTyping

    :param method: The name of the registered censor method to be used.
    :type method: str

    :param areas: A list of tuples representing the rectangular areas to be censored
        in the format ``(x0, y0, x1, y1)``.
    :type areas: List[Tuple[float, float, float, float]]

    :param kwargs: Additional keyword arguments to be passed to the censor method.

    :return: An instance of PIL Image with the censored areas.
    :rtype: Image.Image
    """
    image = load_image(image, mode='RGB')
    c = _get_censor_instance(method)
    for x0, y0, x1, y1 in areas:
        image = c.censor_area(image, (int(x0), int(y0), int(x1), int(y1)), **kwargs)

    return image


def censor_nsfw(image: ImageTyping, method: str, nipple_f: bool = False, penis: bool = True, pussy: bool = True,
                level: str = 's', version: str = 'v1.0', max_infer_size=640,
                conf_threshold: float = 0.3, iou_threshold: float = 0.7, **kwargs):
    """
    Applies censoring to sensitive areas in NSFW images based on object detection.

    :param image: The input image to be censored.
    :type image: ImageTyping

    :param method: The name of the registered censor method to be used.
    :type method: str

    :param nipple_f: Whether to censor female nipples. Default is ``False``.
    :type nipple_f: bool

    :param penis: Whether to censor penises. Default is ``True``.
    :type penis: bool

    :param pussy: Whether to censor vaginas. Default is ``True``.
    :type pussy: bool

    :param level: The scale for NSFW object detection model.
        Options are ``s`` (small), ``n`` (nano, faster than ``s``). Default is ``s``.
    :type level: str

    :param version: The version of the NSFW object detection model. Default is ``v1.0``.
    :type version: str

    :param max_infer_size: The maximum size for inference. Default is ``640``.
    :type max_infer_size: int

    :param conf_threshold: The confidence threshold for object detection. Default is ``0.3``.
    :type conf_threshold: float

    :param iou_threshold: The IoU (Intersection over Union) threshold for non-maximum suppression. Default is ``0.7``.
    :type iou_threshold: float

    :param kwargs: Additional keyword arguments to be passed to the censor method.

    :return: An instance of PIL Image with the sensitive areas censored.
    :rtype: Image.Image
    """
    image = load_image(image, mode='RGB')
    areas = detect_censors(image, level, version, max_infer_size, conf_threshold, iou_threshold)

    c_areas = []
    for area, label, score in areas:
        if (label == 'nipple_f' and nipple_f) or (label == 'penis' and penis) or (label == 'pussy' and pussy):
            c_areas.append(area)

    return censor_areas(image, method, c_areas, **kwargs)
