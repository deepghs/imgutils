"""
Overview:
    A tool for obscuring specified regions on an image.
"""
from typing import Tuple, Type, List, Optional

from PIL import Image, ImageFilter

from ..data import ImageTyping, load_image
from ..detect import detect_censors
from ..utils import ts_lru_cache


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
            >>> pixelate_4 = censor_areas(image, 'pixelate', areas)
            >>>
            >>> # radius=8
            >>> pixelate_8 = censor_areas(image, 'pixelate', areas, radius=8)
            >>>
            >>> # radius=12
            >>> pixelate_12 = censor_areas(image, 'pixelate', areas, radius=12)

            This is the result:

            .. image:: censor_pixelate.plot.py.svg
                :align: center
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
            >>> blur_4 = censor_areas(image, 'blur', areas)
            >>>
            >>> # radius=8
            >>> blur_8 = censor_areas(image, 'blur', areas, radius=8)
            >>>
            >>> # radius=12
            >>> blur_12 = censor_areas(image, 'blur', areas, radius=12)

            This is the result:

            .. image:: censor_blur.plot.py.svg
                :align: center
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
            >>> color_default = censor_areas(image, 'color', areas)
            >>>
            >>> # green
            >>> color_green = censor_areas(image, 'color', areas, color='green')
            >>>
            >>> # #ffff00
            >>> color_ffff00 = censor_areas(image, 'color', areas, color='#ffff00')

            This is the result:

            .. image:: censor_color.plot.py.svg
                :align: center
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


@ts_lru_cache()
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
        >>> # censor with black color
        >>> color_black = censor_areas(origin, 'color', areas, color='black')
        >>>
        >>> # censor with pixelate
        >>> pixelate = censor_areas(origin, 'pixelate', areas, radius=12)
        >>>
        >>> # censor with emoji
        >>> emoji = censor_areas(origin, 'emoji', areas)

        This is the result:

        .. image:: censor_areas.plot.py.svg
            :align: center

    """
    image = load_image(image, mode='RGB')
    c = _get_censor_instance(method)
    for x0, y0, x1, y1 in areas:
        image = c.censor_area(image, (int(x0), int(y0), int(x1), int(y1)), **kwargs)

    return image


def censor_nsfw(image: ImageTyping, method: str, nipple_f: bool = False, penis: bool = True, pussy: bool = True,
                level: str = 's', version: str = 'v1.0', model_name: Optional[str] = None,
                conf_threshold: float = 0.3, iou_threshold: float = 0.7, **kwargs):
    """
    Applies censoring to sensitive areas in NSFW images based on object detection.

    The censor area selected by this function is provided by the :func:`imgutils.detect.censor.detect_censors` function.

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

    :param model_name: Optional custom model name. If not provided, it will be constructed
                       from the version and level.
    :type model_name: Optional[str]

    :param conf_threshold: The confidence threshold for object detection. Default is ``0.3``.
    :type conf_threshold: float

    :param iou_threshold: The IoU (Intersection over Union) threshold for non-maximum suppression. Default is ``0.7``.
    :type iou_threshold: float

    :param kwargs: Additional keyword arguments to be passed to the censor method.

    :return: An instance of PIL Image with the sensitive areas censored.
    :rtype: Image.Image

    Examples::
        >>> from PIL import Image
        >>> from imgutils.operate import censor_nsfw
        >>>
        >>> origin = Image.open('nude_girl.png')
        >>>
        >>> # censor with black color
        >>> color_black = censor_nsfw(origin, 'color', nipple_f=True, color='black')
        >>>
        >>> # censor with pixelate
        >>> pixelate = censor_nsfw(origin, 'pixelate', nipple_f=True, radius=12)
        >>>
        >>> # censor with emoji
        >>> emoji = censor_nsfw(origin, 'emoji', nipple_f=True)

        .. collapse:: This is the result (Warning: NSFW!!!)

            .. image:: censor_nsfw.plot.py.svg
                :align: center

    """
    image = load_image(image, mode='RGB')
    areas = detect_censors(
        image=image,
        level=level,
        version=version,
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    c_areas = []
    for area, label, score in areas:
        if (label == 'nipple_f' and nipple_f) or (label == 'penis' and penis) or (label == 'pussy' and pussy):
            c_areas.append(area)

    return censor_areas(image, method, c_areas, **kwargs)
