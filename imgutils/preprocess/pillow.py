"""
This module provides utilities for image processing using the PIL library, allowing for transformations such as resizing,
cropping, converting to tensor, normalization, and more. It supports both basic and composite transformations, which can
be registered and parsed dynamically. The module also offers functionality to convert images to tensors and normalize them,
making it suitable for preprocessing tasks in machine learning applications.

Transforms can be applied individually or composed into sequences using the PillowCompose class, which applies a list of
transforms sequentially to an image. Each transformation can be registered with decorators to facilitate dynamic creation
and parsing based on configuration dictionaries or lists, supporting flexible and configurable preprocessing pipelines.
"""

import copy
import io
from functools import wraps
from textwrap import indent
from typing import Union, Optional, Tuple, List

import numpy as np
from PIL import Image

from .base import NotParseTarget
from ..data import load_image

# noinspection PyUnresolvedReferences
_INT_TO_PILLOW = {
    0: Image.NEAREST,
    2: Image.BILINEAR,
    3: Image.BICUBIC,
    4: Image.BOX,
    5: Image.HAMMING,
    1: Image.LANCZOS
}

# noinspection PyUnresolvedReferences
_STR_TO_PILLOW = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'box': Image.BOX,
    'hamming': Image.HAMMING,
    'lanczos': Image.LANCZOS
}
_PILLOW_TO_STR = {
    value: key
    for key, value in _STR_TO_PILLOW.items()
}


def _get_pillow_resample(value: Union[int, str]) -> int:
    """
    Converts a resampling filter name or code to a Pillow resampling filter.

    :param value: The resampling filter name (str) or code (int).
    :type value: Union[int, str]
    :return: The corresponding Pillow resampling filter.
    :rtype: int
    :raises ValueError: If the provided value is not a recognized resampling filter.
    :raises TypeError: If the type of `value` is neither int nor str.
    """
    if isinstance(value, int):
        if value not in _INT_TO_PILLOW:
            raise ValueError(f'Invalid interpolation value - {value!r}.')
        return _INT_TO_PILLOW[value]
    elif isinstance(value, str):
        value = value.lower()
        if value not in _STR_TO_PILLOW:
            raise ValueError(f'Invalid interpolation value - {value!r}.')
        return _STR_TO_PILLOW[value]
    else:
        raise TypeError(f"Input type must be int or str, got {type(value)}")


_PTRANS_CREATORS = {}


def register_pillow_transform(name: str):
    """
    Decorator to register a function as a creator for a specific type of Pillow transform.

    :param name: The name of the transform.
    :type name: str
    """

    def _fn(func):
        _PTRANS_CREATORS[name] = func
        return func

    return _fn


_PTRANS_PARSERS = {}


def register_pillow_parse(name: str):
    """
    Decorator to register a function as a parser for a specific type of Pillow transform.

    :param name: The name of the transform.
    :type name: str
    """

    def _fn(func):
        @wraps(func)
        def _new_func(*args, **kwargs):
            return {
                'type': name,
                **func(*args, **kwargs),
            }

        _PTRANS_PARSERS[name] = _new_func
        return _new_func

    return _fn


class PillowResize:
    """
    A class to resize an image to a specified size using Pillow.

    :param size: Target size. If an int, the smallest side is matched to this number maintaining aspect ratio.
                 If a tuple or list of two ints, size is matched to these dimensions (width, height).
    :type size: Union[int, List[int], Tuple[int, ...]]
    :param interpolation: Resampling filter for resizing.
    :type interpolation: int
    :param max_size: If provided, resizes the image such that the maximum size does not exceed this value.
    :type max_size: Optional[int]
    :param antialias: Whether to apply an anti-aliasing filter.
    :type antialias: bool
    :raises TypeError: If `size` is not int, list, or tuple.
    :raises ValueError: If `size` is a list or tuple but does not contain 1 or 2 elements.
    """

    # noinspection PyUnresolvedReferences
    def __init__(
            self,
            size: Union[int, List[int], Tuple[int, ...]],
            interpolation: int = Image.BILINEAR,
            max_size: Optional[int] = None,
            antialias: bool = True
    ):
        if not isinstance(size, (int, list, tuple)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, (list, tuple)) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        if max_size is not None and isinstance(size, (list, tuple)) and len(size) != 1:
            raise ValueError("max_size is only supported for single int size or sequence of length 1")

        self.size = size
        self.interpolation = interpolation
        # noinspection PyTypeChecker
        self.max_size = max_size
        self.antialias = antialias

    def _get_resize_size(self, img: Image.Image) -> Tuple[int, int]:
        """
        Calculate the target resize dimensions for the image.

        :param img: The image to resize.
        :type img: Image.Image
        :return: The calculated dimensions to resize to.
        :rtype: Tuple[int, int]
        """
        w, h = img.size
        if isinstance(self.size, int) or (isinstance(self.size, (list, tuple)) and len(self.size) == 1):
            size = self.size if isinstance(self.size, int) else self.size[0]
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            if self.max_size is not None:
                max_size = self.max_size
                # noinspection PyTypeChecker
                if max(oh, ow) > max_size:
                    if oh > ow:
                        # noinspection PyTypeChecker
                        ow = int(max_size * ow / oh)
                        oh = max_size
                    else:
                        # noinspection PyTypeChecker
                        oh = int(max_size * oh / ow)
                        ow = max_size

            return ow, oh
        else:
            # noinspection PyUnresolvedReferences
            return self.size[1], self.size[0]

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Resize the image according to the specified parameters.

        :param img: The image to resize.
        :type img: Image.Image
        :return: The resized image.
        :rtype: Image.Image
        :raises TypeError: If the input is not an image.
        """
        if not isinstance(img, Image.Image):
            raise TypeError('Input must be a PIL Image')

        size = self._get_resize_size(img)
        width, height = size
        if width != img.width or height != img.height:
            # noinspection PyUnresolvedReferences
            if self.interpolation in {Image.BILINEAR, Image.BICUBIC}:
                return img.resize(size, self.interpolation, reducing_gap=None if self.antialias else 1.0)
            else:
                return img.resize(size, self.interpolation)
        else:
            return img

    def __repr__(self) -> str:
        """
        Represent the PillowResize instance as a string.

        :return: String representation of the instance.
        :rtype: str
        """
        interpolate_str = _PILLOW_TO_STR[self.interpolation]
        detail = f"(size={self.size}, interpolation={interpolate_str}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


@register_pillow_transform('resize')
def _create_resize(size, interpolation='bilinear', max_size=None, antialias=True):
    """
    Create a PillowResize transformation.

    :param size: Target size for resizing, either an int or a tuple/list of two ints.
    :type size: Union[int, Tuple[int, int], List[int]]
    :param interpolation: The interpolation method to use.
    :type interpolation: str
    :param max_size: Optional maximum size to ensure the image does not exceed.
    :type max_size: Optional[int]
    :param antialias: Whether to apply anti-aliasing.
    :type antialias: bool
    :return: An instance of PillowResize.
    :rtype: PillowResize
    """
    # noinspection PyTypeChecker
    return PillowResize(
        size=size,
        interpolation=_get_pillow_resample(interpolation),
        max_size=max_size,
        antialias=antialias,
    )


@register_pillow_parse('resize')
def _parse_resize(obj: PillowResize):
    """
    Parse a PillowResize object to a dictionary representing its configuration.

    :param obj: The PillowResize object to parse.
    :type obj: PillowResize
    :return: A dictionary containing the configuration of the PillowResize object.
    :rtype: dict
    :raises NotParseTarget: If the object is not an instance of PillowResize.
    """
    if not isinstance(obj, PillowResize):
        raise NotParseTarget

    return {
        'size': obj.size,
        'interpolation': _PILLOW_TO_STR[obj.interpolation],
        'max_size': obj.max_size,
        'antialias': obj.antialias,
    }


class PillowCenterCrop:
    """
    A class for center cropping an image using Pillow.

    :param size: Desired output size of the crop. If an int is provided, the crop will be a square of that size.
                 If a tuple or list of two ints is provided, it specifies the size as (height, width).
    :type size: Union[int, Tuple[int, int], List[int]]
    :raises ValueError: If `size` is neither an int nor a tuple/list of two ints.
    """

    def __init__(self, size):
        """
        Initialize the PillowCenterCrop instance.

        :param size: The target size for cropping.
        """
        if isinstance(size, int):
            # noinspection PyTypeChecker
            self.size = (size, size)
        elif isinstance(size, (tuple, list)) and len(size) == 1:
            # noinspection PyTypeChecker
            self.size = (size[0], size[0])
        elif isinstance(size, (tuple, list)) and len(size) == 2:
            # noinspection PyTypeChecker
            self.size = (size[0], size[1])
        else:
            raise ValueError("Please provide only two dimensions (h, w) for size.")

    def __call__(self, img):
        """
        Apply a center crop to the image.

        :param img: The image to crop.
        :type img: Image.Image
        :return: The cropped image.
        :rtype: Image.Image
        :raises TypeError: If the input is not an image.
        """
        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image')

        return self._center_crop(img)

    def _center_crop(self, img):
        """
        Perform the actual center cropping operation on the image.

        :param img: The image to crop.
        :type img: Image.Image
        :return: The cropped image.
        :rtype: Image.Image
        """
        width, height = img.size
        crop_height, crop_width = self.size

        if width < crop_width or height < crop_height:
            pad_width = max(crop_width - width, 0)
            pad_height = max(crop_height - height, 0)

            left = pad_width // 2
            top = pad_height // 2

            new_width = width + pad_width
            new_height = height + pad_height
            new_img = Image.new(img.mode, (new_width, new_height), (0, 0, 0))
            new_img.paste(img, (left, top))

            img = new_img
            width, height = img.size

        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        return img.crop((left, top, right, bottom))

    def __repr__(self) -> str:
        """
        Represent the PillowCenterCrop instance as a string.

        :return: String representation of the instance.
        :rtype: str
        """
        return f"{self.__class__.__name__}(size={self.size})"


@register_pillow_transform('center_crop')
def _create_center_crop(size):
    """
    Create a PillowCenterCrop transformation.

    :param size: Target size for cropping, either an int or a tuple/list of two ints.
    :type size: Union[int, Tuple[int, int], List[int]]
    :return: An instance of PillowCenterCrop.
    :rtype: PillowCenterCrop
    """
    return PillowCenterCrop(
        size=size
    )


@register_pillow_parse('center_crop')
def _parse_center_crop(obj: PillowCenterCrop):
    """
    Parse a PillowCenterCrop object to a dictionary representing its configuration.

    :param obj: The PillowCenterCrop object to parse.
    :type obj: PillowCenterCrop
    :return: A dictionary containing the configuration of the PillowCenterCrop object.
    :rtype: dict
    :raises NotParseTarget: If the object is not an instance of PillowCenterCrop.
    """
    if not isinstance(obj, PillowCenterCrop):
        raise NotParseTarget

    return {
        'size': list(obj.size),
    }


class PillowToTensor:
    """
    A class to convert a PIL Image to a tensor, handling different image modes appropriately.
    """

    def __call__(self, pic):
        """
        Convert a PIL Image to a tensor.

        :param pic: The picture to convert.
        :type pic: Image.Image
        :return: The picture as a numpy array with dimensions depending on the image mode.
        :rtype: np.ndarray
        :raises TypeError: If the input is not a PIL Image.
        :raises ValueError: If the image mode is not supported.
        """
        if not isinstance(pic, Image.Image):
            raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

        if pic.mode == 'I':
            # 32-bit signed integer pixels
            return np.array(pic, np.int32, copy=True)[None, ...]
        elif pic.mode == 'I;16':
            # 16-bit signed integer pixels
            return np.array(pic, np.int16, copy=True)[None, ...]
        elif pic.mode == 'F':
            # 32-bit floating point pixels
            return np.array(pic, np.float32, copy=True)[None, ...]

        img = np.array(pic, copy=True)
        if pic.mode == '1':
            return img.astype(np.float32)[None, ...]
        elif pic.mode == 'L':
            img = img.reshape((1,) + img.shape)
            return img.astype(np.float32) / 255
        elif pic.mode == 'LA':
            img_l = img[..., 0].reshape((1,) + img.shape[:2])
            img_a = img[..., 1].reshape((1,) + img.shape[:2])
            return np.concatenate((img_l, img_a), axis=0).astype(np.float32) / 255
        elif pic.mode == 'P':
            img = np.array(pic, copy=True)[None, ...]
            return img.astype(np.float32) / 255
        elif pic.mode in ('RGB', 'YCbCr'):
            img = img.transpose((2, 0, 1))
            return img.astype(np.float32) / 255
        elif pic.mode == 'RGBA':
            img = img.transpose((2, 0, 1))
            return img.astype(np.float32) / 255
        elif pic.mode == 'CMYK':
            img = img.transpose((2, 0, 1))
            return img.astype(np.float32) / 255

        raise ValueError(f"Unsupported PIL image mode: {pic.mode}")  # pragma: no cover

    def __repr__(self) -> str:
        """
        Represent the PillowToTensor instance as a string.

        :return: String representation of the instance.
        :rtype: str
        """
        return f"{self.__class__.__name__}()"


@register_pillow_transform('to_tensor')
def _create_to_tensor():
    """
    Create a PillowToTensor transformation.

    :return: An instance of PillowToTensor.
    :rtype: PillowToTensor
    """
    return PillowToTensor()


@register_pillow_parse('to_tensor')
def _parse_to_tensor(obj: PillowToTensor):
    """
    Parse a PillowToTensor object to a dictionary representing its configuration.

    :param obj: The PillowToTensor object to parse.
    :type obj: PillowToTensor
    :return: A dictionary representing the configuration of the PillowToTensor object.
    :rtype: dict
    :raises NotParseTarget: If the object is not an instance of PillowToTensor.
    """
    if not isinstance(obj, PillowToTensor):
        raise NotParseTarget

    return {}


class PillowMaybeToTensor:
    """
    A class to conditionally convert an image or numpy array to a tensor.
    """

    def __call__(self, image):
        """
        Convert an image to a tensor if it is not already a numpy array.

        :param image: The image to potentially convert.
        :type image: Union[Image.Image, np.ndarray]
        :return: The image as a numpy array.
        :rtype: np.ndarray
        """
        if isinstance(image, np.ndarray):
            return image
        else:
            return PillowToTensor()(image)

    def __repr__(self):
        """
        Represent the PillowMaybeToTensor instance as a string.

        :return: String representation of the instance.
        :rtype: str
        """
        return f'{type(self).__name__}()'


@register_pillow_transform('maybe_to_tensor')
def _create_maybe_to_tensor():
    """
    Create a PillowMaybeToTensor transformation.

    :return: An instance of PillowMaybeToTensor.
    :rtype: PillowMaybeToTensor
    """
    return PillowMaybeToTensor()


@register_pillow_parse('maybe_to_tensor')
def _parse_maybe_to_tensor(obj: PillowMaybeToTensor):
    """
    Parse a PillowMaybeToTensor object to a dictionary representing its configuration.

    :param obj: The PillowMaybeToTensor object to parse.
    :type obj: PillowMaybeToTensor
    :return: A dictionary representing the configuration of the PillowMaybeToTensor object.
    :rtype: dict
    :raises NotParseTarget: If the object is not an instance of PillowMaybeToTensor.
    """
    if not isinstance(obj, PillowMaybeToTensor):
        raise NotParseTarget

    return {}


class PillowNormalize:
    """
    Normalizes an image by subtracting the mean and dividing by the standard deviation.

    :param mean: The mean value(s) for normalization.
    :param std: The standard deviation value(s) for normalization.
    :param inplace: If True, perform normalization in-place.
    :type inplace: bool
    """

    def __init__(self, mean, std, inplace: bool = False):
        if isinstance(mean, (list, tuple)):
            self.mean = np.array(mean, dtype=np.float32)
        else:
            self.mean = np.array([float(mean)], dtype=np.float32)
        if isinstance(std, (list, tuple)):
            self.std = np.array(std, dtype=np.float32)
        else:
            self.std = np.array([float(std)], dtype=np.float32)
        self.inplace = inplace

    def __call__(self, array):
        """
        Apply normalization to the input array.

        :param array: The input image array.
        :type array: np.ndarray
        :return: The normalized image array.
        :rtype: np.ndarray
        :raises TypeError: If the input is not a numpy.ndarray or is not a float array.
        :raises ValueError: If the input array does not have the expected dimensions.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError('Input should be a numpy.ndarray')
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f'Input tensor should ba a float array, but {array.dtype!r} given.')

        if array.ndim < 3:
            raise ValueError(f'Expected array to be an array image of size (..., C, H, W). '
                             f'Got array.shape == {array.shape!r}')
        else:
            if not self.inplace:
                array = array.copy()
            return self._normalize_multi(array)

    def _normalize_multi(self, array):
        """
        Helper function to normalize each channel of the image.

        :param array: The input image array.
        :type array: np.ndarray
        :return: The normalized image array.
        :rtype: np.ndarray
        """
        mean = self.mean.reshape(-1, 1, 1)
        std = self.std.reshape(-1, 1, 1)
        array -= mean
        array /= std
        return array

    def __repr__(self) -> str:
        """
        String representation of the PillowNormalize instance.

        :return: String representation.
        :rtype: str
        """
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


@register_pillow_transform('normalize')
def _create_normalize(mean, std, inplace=False):
    """
    Factory function to create a PillowNormalize instance.

    :param mean: Mean value(s) for normalization.
    :type mean: np.ndarray
    :param std: Standard deviation value(s) for normalization.
    :type std: np.ndarray
    :param inplace: Perform normalization in-place.
    :type inplace: bool
    :return: An instance of PillowNormalize.
    :rtype: PillowNormalize
    """
    return PillowNormalize(
        mean=mean,
        std=std,
        inplace=inplace,
    )


@register_pillow_parse('normalize')
def _parse_normalize(obj: PillowNormalize):
    """
    Parse a PillowNormalize instance into a serializable dictionary.

    :param obj: The PillowNormalize instance to parse.
    :type obj: PillowNormalize
    :return: A dictionary with mean and std values.
    :rtype: dict
    :raises NotParseTarget: If the object is not a PillowNormalize instance.
    """
    if not isinstance(obj, PillowNormalize):
        raise NotParseTarget

    return {
        'mean': obj.mean.tolist(),
        'std': obj.std.tolist(),
    }


class PillowConvertRGB:
    """
    A class for converting images to RGB format.

    This class provides functionality to convert PIL Images to RGB format,
    with an option to specify background color for images with transparency.

    :param force_background: Background color to use when converting images with alpha channel.
                           Default is 'white'.
    """

    def __init__(self, force_background: Optional[str] = 'white'):
        self.force_background = force_background

    def __call__(self, pic):
        """
        Convert the input image to RGB format.

        :param pic: Input image to be converted
        :type pic: PIL.Image.Image

        :return: RGB converted image
        :rtype: PIL.Image.Image
        :raises TypeError: If input is not a PIL Image
        """
        if not isinstance(pic, Image.Image):
            raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))
        return load_image(pic, mode='RGB', force_background=self.force_background)

    def __repr__(self):
        """
        Return string representation of the class.

        :return: String representation
        :rtype: str
        """
        return f'{self.__class__.__name__}(force_background={self.force_background!r})'


@register_pillow_transform('convert_rgb')
def _create_convert_rgb(force_background: Optional[str] = 'white'):
    """
    Factory function to create PillowConvertRGB instance.

    :param force_background: Background color for transparency conversion
    :type force_background: Optional[str]

    :return: PillowConvertRGB instance
    :rtype: PillowConvertRGB
    """
    return PillowConvertRGB(force_background=force_background)


@register_pillow_parse('convert_rgb')
def _parse_convert_rgb(obj):
    """
    Parse PillowConvertRGB object to dictionary configuration.

    :param obj: Object to parse
    :type obj: Any

    :return: Configuration dictionary
    :rtype: dict
    :raises NotParseTarget: If object is not PillowConvertRGB instance
    """
    if not isinstance(obj, PillowConvertRGB):
        raise NotParseTarget

    obj: PillowConvertRGB
    return {
        'force_background': obj.force_background,
    }


class PillowRescale:
    """
    A class for rescaling image pixel values.

    This class provides functionality to rescale numpy array values by a given factor,
    commonly used to normalize image pixel values (e.g., from [0-255] to [0-1]).

    :param rescale_factor: Factor to multiply pixel values by. Default is 1/255.
    :type rescale_factor: float
    """

    def __init__(self, rescale_factor: float = 1 / 255):
        self.rescale_factor = np.float32(rescale_factor)

    def __call__(self, array):
        """
        Rescale the input array values.

        :param array: Input array to be rescaled
        :type array: numpy.ndarray

        :return: Rescaled array
        :rtype: numpy.ndarray
        :raises TypeError: If input is not a numpy array
        """
        if not isinstance(array, np.ndarray):
            raise TypeError('Input should be a numpy.ndarray')
        return array * self.rescale_factor

    def __repr__(self):
        """
        Return string representation of the class.

        :return: String representation
        :rtype: str
        """
        return f'{self.__class__.__name__}(rescale_factor={self.rescale_factor!r})'


@register_pillow_transform('rescale')
def _create_rescale(rescale_factor: float = 1 / 255):
    """
    Factory function to create PillowRescale instance.

    :param rescale_factor: Factor for rescaling pixel values
    :type rescale_factor: float

    :return: PillowRescale instance
    :rtype: PillowRescale
    """
    return PillowRescale(rescale_factor=rescale_factor)


@register_pillow_parse('rescale')
def _parse_rescale(obj):
    """
    Parse PillowRescale object to dictionary configuration.

    :param obj: Object to parse
    :type obj: Any

    :return: Configuration dictionary
    :rtype: dict
    :raises NotParseTarget: If object is not PillowRescale instance
    """
    if not isinstance(obj, PillowRescale):
        raise NotParseTarget

    obj: PillowRescale
    return {
        'rescale_factor': obj.rescale_factor.item(),
    }


class PillowCompose:
    """
    Composes several transforms together into a single transform.

    :param transforms: A list of transformations to compose.
    :type transforms: list
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        """
        Apply the composed transformations to an image.

        :param image: The input image.
        :type image: Any
        :return: The transformed image.
        :rtype: Any
        """
        x = image
        for trans in self.transforms:
            x = trans(x)
        return x

    def __repr__(self):
        """
        String representation of the PillowCompose instance.

        :return: String representation.
        :rtype: str
        """
        with io.StringIO() as sf:
            print(f'{type(self).__name__}(', file=sf)
            for trans in self.transforms:
                print(indent(repr(trans), prefix='    '), file=sf)
            print(f')', file=sf)
            return sf.getvalue()


def create_pillow_transforms(tvalue: Union[list, dict]):
    """
    Create a transformation or a composition of transformations based on the input value.

    :param tvalue: A list or dictionary describing the transformation(s).
    :type tvalue: Union[list, dict]
    :return: A transformation or a composition of transformations.
    :rtype: Union[PillowCompose, Any]
    :raises TypeError: If the input value is not a list or dictionary.

    :example:
        >>> from imgutils.preprocess import create_pillow_transforms
        >>>
        >>> create_pillow_transforms({
        ...     'type': 'resize',
        ...     'size': 384,
        ...     'interpolation': 'bicubic',
        ... })
        PillowResize(size=384, interpolation=bicubic, max_size=None, antialias=True)
        >>> create_pillow_transforms({
        ...     'type': 'resize',
        ...     'size': (224, 256),
        ...     'interpolation': 'bilinear',
        ... })
        PillowResize(size=(224, 256), interpolation=bilinear, max_size=None, antialias=True)
        >>> create_pillow_transforms({'type': 'center_crop', 'size': 224})
        PillowCenterCrop(size=(224, 224))
        >>> create_pillow_transforms({'type': 'to_tensor'})
        PillowToTensor()
        >>> create_pillow_transforms({'type': 'maybe_to_tensor'})
        PillowMaybeToTensor()
        >>> create_pillow_transforms({'type': 'normalize', 'mean': 0.5, 'std': 0.5})
        PillowNormalize(mean=[0.5], std=[0.5])
        >>> create_pillow_transforms({
        ...     'type': 'normalize',
        ...     'mean': [0.485, 0.456, 0.406],
        ...     'std': [0.229, 0.224, 0.225],
        ... })
        PillowNormalize(mean=[0.485 0.456 0.406], std=[0.229 0.224 0.225])
        >>> create_pillow_transforms([
        ...     {'antialias': True,
        ...      'interpolation': 'bicubic',
        ...      'max_size': None,
        ...      'size': 384,
        ...      'type': 'resize'},
        ...     {'size': (224, 224), 'type': 'center_crop'},
        ...     {'type': 'maybe_to_tensor'},
        ...     {'mean': 0.5, 'std': 0.5, 'type': 'normalize'}
        ... ])
        PillowCompose(
            PillowResize(size=384, interpolation=bicubic, max_size=None, antialias=True)
            PillowCenterCrop(size=(224, 224))
            PillowMaybeToTensor()
            PillowNormalize(mean=[0.5], std=[0.5])
        )
    """
    if isinstance(tvalue, list):
        return PillowCompose([create_pillow_transforms(titem) for titem in tvalue])
    elif isinstance(tvalue, dict):
        tvalue = copy.deepcopy(tvalue)
        ttype = tvalue.pop('type')
        return _PTRANS_CREATORS[ttype](**tvalue)
    else:
        raise TypeError(f'Unknown type of transforms - {tvalue!r}.')


def parse_pillow_transforms(value):
    """
    Parse transformations into a serializable format.

    :param value: The transformation or composition to parse.
    :type value: Any
    :return: A serializable representation of the transformation.
    :rtype: Union[list, dict]
    :raises TypeError: If the transformation cannot be parsed.

    :example:
        >>> from PIL import Image
        >>>
        >>> from imgutils.preprocess import parse_pillow_transforms
        >>> from imgutils.preprocess.pillow import PillowResize, PillowCenterCrop, PillowMaybeToTensor, PillowToTensor, \
        ...     PillowNormalize
        >>>
        >>> parse_pillow_transforms(PillowResize(
        ...     size=384,
        ...     interpolation=Image.BICUBIC,
        ... ))
        {'type': 'resize', 'size': 384, 'interpolation': 'bicubic', 'max_size': None, 'antialias': True}
        >>> parse_pillow_transforms(PillowResize(
        ...     size=(224, 256),
        ...     interpolation=Image.BILINEAR,
        ... ))
        {'type': 'resize', 'size': (224, 256), 'interpolation': 'bilinear', 'max_size': None, 'antialias': True}
        >>> parse_pillow_transforms(PillowCenterCrop(size=224))
        {'type': 'center_crop', 'size': [224, 224]}
        >>> parse_pillow_transforms(PillowToTensor())
        {'type': 'to_tensor'}
        >>> parse_pillow_transforms(PillowMaybeToTensor())
        {'type': 'maybe_to_tensor'}
        >>> parse_pillow_transforms(PillowNormalize(mean=0.5, std=0.5))
        {'type': 'normalize', 'mean': [0.5], 'std': [0.5]}
        >>> parse_pillow_transforms(PillowNormalize(
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225],
        ... ))
        {'type': 'normalize', 'mean': [0.48500001430511475, 0.4560000002384186, 0.4059999883174896], 'std': [0.2290000021457672, 0.2240000069141388, 0.22499999403953552]}
        >>> parse_pillow_transforms(PillowCompose([
        ...     PillowResize(
        ...         size=384,
        ...         interpolation=Image.BICUBIC,
        ...     ),
        ...     PillowCenterCrop(size=224),
        ...     PillowMaybeToTensor(),
        ...     PillowNormalize(mean=0.5, std=0.5),
        ... ]))
        [{'antialias': True,
          'interpolation': 'bicubic',
          'max_size': None,
          'size': 384,
          'type': 'resize'},
         {'size': [224, 224], 'type': 'center_crop'},
         {'type': 'maybe_to_tensor'},
         {'mean': [0.5], 'std': [0.5], 'type': 'normalize'}]
    """
    if isinstance(value, PillowCompose):
        return [parse_pillow_transforms(trans) for trans in value.transforms]
    else:
        for key, parser in _PTRANS_PARSERS.items():
            try:
                return parser(value)
            except NotParseTarget:
                pass

        raise TypeError(f'Unknown parse transform - {value!r}.')
