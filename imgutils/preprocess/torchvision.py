"""
This module provides utilities for creating and parsing torchvision transforms.
It includes functionality for registering custom transforms, handling interpolation modes,
and converting between different transform representations.

The module supports common image transformations like resize, center crop, tensor conversion
and normalization. It provides a flexible framework for extending with additional transforms.
"""

import copy
from functools import wraps
from typing import Union, Tuple

from PIL import Image

from .base import NotParseTarget
from ..data import pad_image_to_size

try:
    import torchvision
    import torch
except (ImportError, ModuleNotFoundError):
    _HAS_TORCHVISION = False
    torchvision = None
    torch = None
else:
    _HAS_TORCHVISION = True


def _check_torchvision():
    """
    Check if torchvision is available and raise error if not installed.

    :raises EnvironmentError: If torchvision is not installed
    """
    if not _HAS_TORCHVISION:
        raise EnvironmentError('No torchvision available.\n'
                               'Please install it by `pip install dghs-imgutils[torchvision]`.')


def _get_interpolation_mode(value):
    """
    Convert different interpolation mode representations to torchvision.transforms.InterpolationMode.

    :param value: The interpolation mode value to convert. Can be int, string or InterpolationMode
    :return: The corresponding InterpolationMode enum value
    :raises ValueError: If the interpolation value is invalid
    :raises TypeError: If the value type is not supported
    """
    from torchvision.transforms import InterpolationMode
    _INT_TO_INTERMODE = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }

    _STR_TO_INTERMODE = {
        value.value: value
        for key, value in InterpolationMode.__members__.items()
    }

    if isinstance(value, InterpolationMode):
        return value
    elif isinstance(value, int):
        if value not in _INT_TO_INTERMODE:
            raise ValueError(f'Invalid interpolation value - {value!r}.')
        return _INT_TO_INTERMODE[value]
    elif isinstance(value, str):
        value = value.lower()
        if value not in _STR_TO_INTERMODE:
            raise ValueError(f'Invalid interpolation value - {value!r}.')
        return _STR_TO_INTERMODE[value]
    else:
        raise TypeError(f'Unknown type of interpolation mode - {value!r}.')


def _get_int_from_interpolation_mode(value):
    """
    Convert a torchvision.transforms.InterpolationMode enum value to its corresponding integer representation.

    This function performs the reverse operation of _get_interpolation_mode, converting
    InterpolationMode enum values back to their integer representations.

    :param value: The InterpolationMode enum value to convert
    :type value: InterpolationMode

    :return: The integer representation of the interpolation mode
    :rtype: int

    :raises TypeError: If the input value is not an InterpolationMode enum value

    :examples:
        >>> mode = InterpolationMode.BILINEAR
        >>> _get_int_from_interpolation_mode(mode)  # Returns 2
    """
    from torchvision.transforms import InterpolationMode
    if not isinstance(value, InterpolationMode):
        raise TypeError(
            f'Unknown type of interpolation mode, cannot be transformed to int - {value!r}')  # pragma: no cover

    _INTERMODE_TO_INT = {
        InterpolationMode.NEAREST: 0,
        InterpolationMode.BILINEAR: 2,
        InterpolationMode.BICUBIC: 3,
        InterpolationMode.BOX: 4,
        InterpolationMode.HAMMING: 5,
        InterpolationMode.LANCZOS: 1,
    }
    return _INTERMODE_TO_INT[value]


def _get_interpolation_str_from_mode(value) -> str:
    """Convert InterpolationMode to string for F.interpolate"""
    from torchvision.transforms import InterpolationMode
    if not isinstance(value, InterpolationMode):
        raise TypeError(
            f'Unknown type of interpolation mode, cannot be transformed to int - {value!r}')  # pragma: no cover

    _INTERMODE_TO_STR = {
        InterpolationMode.NEAREST: 'nearest',
        InterpolationMode.BILINEAR: 'bilinear',
        InterpolationMode.BICUBIC: 'bicubic',
        # For modes not directly supported by F.interpolate, we map to the closest equivalent
        InterpolationMode.BOX: 'area',  # BOX is similar to area interpolation
        InterpolationMode.HAMMING: 'bilinear',  # No direct equivalent, use bilinear
        InterpolationMode.LANCZOS: 'bicubic',  # No direct equivalent, use bicubic
    }
    return _INTERMODE_TO_STR[value]


_TRANS_CREATORS = {}


def _register_transform(name: str, safe: bool = True):
    """
    Register a transform creation function.

    :param name: Name of the transform
    :param safe: Whether to check torchvision availability
    :return: Decorator function
    """
    if safe:
        _check_torchvision()

    def _fn(func):
        _TRANS_CREATORS[name] = func
        return func

    return _fn


def register_torchvision_transform(name: str):
    """
    Register a torchvision transform creation function.

    :param name: Name of the transform
    :return: Decorator function
    """
    return _register_transform(name, safe=True)


_TRANS_PARSERS = {}


def _register_parse(name: str, safe: bool = True):
    """
    Register a transform parsing function.

    :param name: Name of the transform parser
    :param safe: Whether to check torchvision availability
    :return: Decorator function
    """
    if safe:
        _check_torchvision()

    def _fn(func):
        @wraps(func)
        def _new_func(*args, **kwargs):
            return {
                'type': name,
                **func(*args, **kwargs),
            }

        _TRANS_PARSERS[name] = _new_func
        return _new_func

    return _fn


def register_torchvision_parse(name: str):
    """
    Register a torchvision transform parsing function.

    :param name: Name of the transform parser
    :return: Decorator function
    """
    return _register_parse(name, safe=True)


@_register_transform('resize', safe=False)
def _create_resize(size, interpolation='bilinear', max_size=None, antialias=True):
    """
    Create a torchvision Resize transform.

    :param size: Target size
    :param interpolation: Interpolation mode
    :param max_size: Maximum size
    :param antialias: Whether to use anti-aliasing
    :return: Resize transform
    """
    from torchvision.transforms import Resize
    return Resize(
        size=size,
        interpolation=_get_interpolation_mode(interpolation),
        max_size=max_size,
        antialias=antialias,
    )


@_register_parse('resize', safe=False)
def _parse_resize(obj):
    """
    Parse a Resize transform object.

    :param obj: Transform object to parse
    :return: Dict containing transform parameters
    :raises NotParseTarget: If obj is not a Resize transform
    """
    from torchvision.transforms import Resize
    if not isinstance(obj, Resize):
        raise NotParseTarget

    obj: Resize
    return {
        'size': obj.size,
        'interpolation': obj.interpolation.value,
        'max_size': obj.max_size,
        'antialias': obj.antialias,
    }


@_register_transform('center_crop', safe=False)
def _create_center_crop(size):
    """
    Create a torchvision CenterCrop transform.

    :param size: Target size for cropping
    :return: CenterCrop transform
    """
    from torchvision.transforms import CenterCrop
    return CenterCrop(
        size=size,
    )


@_register_parse('center_crop', safe=False)
def _parse_center_crop(obj):
    """
    Parse a CenterCrop transform object.

    :param obj: Transform object to parse
    :return: Dict containing transform parameters
    :raises NotParseTarget: If obj is not a CenterCrop transform
    """
    from torchvision.transforms import CenterCrop
    if not isinstance(obj, CenterCrop):
        raise NotParseTarget

    obj: CenterCrop
    return {
        'size': obj.size,
    }


if _HAS_TORCHVISION:
    from torchvision.transforms import ToTensor


    class MaybeToTensor(ToTensor):
        def __init__(self) -> None:
            super().__init__()

        def __call__(self, pic):
            import torchvision.transforms.functional as F
            import torch
            if isinstance(pic, torch.Tensor):
                return pic
            return F.to_tensor(pic)

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}()"

else:
    MaybeToTensor = None


@_register_transform('maybe_to_tensor', safe=False)
def _create_maybe_to_tensor():
    """
    Create a MaybeToTensor transform that converts input to tensor if not already a tensor.

    :return: MaybeToTensor transform
    """
    return MaybeToTensor()


@_register_parse('maybe_to_tensor', safe=False)
def _parse_maybe_to_tensor(obj):
    """
    Parse a MaybeToTensor transform object.

    :param obj: Transform object to parse
    :return: Empty dict since no parameters needed
    :raises NotParseTarget: If obj is not a MaybeToTensor transform
    """
    if type(obj).__name__ != 'MaybeToTensor':
        raise NotParseTarget

    return {}


@_register_transform('to_tensor', safe=False)
def _create_to_tensor():
    """
    Create a torchvision ToTensor transform.

    :return: ToTensor transform
    """
    from torchvision.transforms import ToTensor
    return ToTensor()


@_register_parse('to_tensor', safe=False)
def _parse_to_tensor(obj):
    """
    Parse a ToTensor transform object.

    :param obj: Transform object to parse
    :return: Empty dict since no parameters needed
    :raises NotParseTarget: If obj is not a ToTensor transform
    """
    if type(obj).__name__ != 'ToTensor':
        raise NotParseTarget

    return {}


@_register_transform('normalize', safe=False)
def _create_normalize(mean, std, inplace=False):
    """
    Create a torchvision Normalize transform.

    :param mean: Sequence of means for each channel
    :param std: Sequence of standard deviations for each channel
    :param inplace: Whether to perform normalization in-place
    :return: Normalize transform
    """
    import torch
    from torchvision.transforms import Normalize
    return Normalize(
        mean=torch.tensor(mean),
        std=torch.tensor(std),
        inplace=inplace,
    )


@_register_parse('normalize', safe=False)
def _parse_normalize(obj):
    """
    Parse a Normalize transform object.

    :param obj: Transform object to parse
    :return: Dict containing transform parameters
    :raises NotParseTarget: If obj is not a Normalize transform
    """
    from torchvision.transforms import Normalize
    if not isinstance(obj, Normalize):
        raise NotParseTarget

    obj: Normalize
    return {
        'mean': obj.mean.tolist() if hasattr(obj.mean, 'tolist') else obj.mean,
        'std': obj.std.tolist() if hasattr(obj.std, 'tolist') else obj.std,
    }


if _HAS_TORCHVISION:
    from torchvision.transforms import InterpolationMode
    import torch.nn.functional as F


    class PadToSize(torch.nn.Module):
        """
        Resize and center-pad PIL image to target size with background color.
        TorchVision-compatible transform that can be composed.

        This transform first resizes the input image to fit within the target size
        while preserving its aspect ratio, then pads the result with the specified
        background color to reach the exact target dimensions.

        :param size: Target size as (height, width) tuple or single int for square output
        :param background_color: Color to use for padding. Can be string name, RGB/RGBA tuple, or single int
        :param interpolation: Interpolation mode for resizing, defaults to BILINEAR
        :type interpolation: InterpolationMode

        :raises ValueError: If size or background_color format is invalid

        :example:

        >>> transform = PadToSize(size=(300, 300), background_color='black')
        >>> padded_image = transform(input_image)
        """

        def __init__(self, size: Union[Tuple[int, int], int],
                     background_color: Union[str, int, Tuple[int, int, int], Tuple[int, int, int, int]] = 'white',
                     interpolation: InterpolationMode = InterpolationMode.BILINEAR):
            super().__init__()
            from ..data.pad import _parse_size, _parse_color_to_rgba
            self.size: Tuple[int, int] = _parse_size(size)
            self.background_color = background_color
            self.interpolation: InterpolationMode = interpolation
            _parse_color_to_rgba(self.background_color)

        def _pad_pil_image(self, pic):
            """
            Pad a PIL image to the target size.

            :param pic: Input PIL Image
            :type pic: PIL.Image.Image

            :return: Padded PIL Image
            :rtype: PIL.Image.Image
            """
            return pad_image_to_size(
                pic=pic,
                size=self.size,
                background_color=self.background_color,
                interpolation=_get_int_from_interpolation_mode(self.interpolation),
            )

        def _pad_tensor(self, tensor):
            """
            Pad a tensor to the target size.

            :param tensor: Input tensor image
            :type tensor: torch.Tensor

            :return: Padded tensor
            :rtype: torch.Tensor
            :raises ValueError: If tensor dimensions are not 3 or 4
            """
            from ..data.pad import _parse_color_to_mode

            if tensor.ndim < 3 or tensor.ndim > 4:
                raise ValueError(f"Tensor should have 3 or 4 dimensions, got {tensor.ndim}")

            # Handle batched and unbatched tensors
            is_batched = tensor.ndim == 4
            if not is_batched:
                tensor = tensor.unsqueeze(0)

            # Get tensor properties
            b, c, h, w = tensor.shape
            target_w, target_h = self.size

            # Calculate new dimensions preserving aspect ratio
            ratio = min(target_w / w, target_h / h)
            new_h, new_w = round(h * ratio), round(w * ratio)

            # Resize tensor
            mode = _get_interpolation_str_from_mode(self.interpolation)
            resized = F.interpolate(
                tensor.type(torch.float32),
                size=(new_h, new_w),
                mode=mode,
                align_corners=None if mode in {'nearest', 'area'} else False,
                antialias=True if mode in {'bicubic', 'bilinear'} else False,
            )
            if tensor.dtype.is_floating_point:
                resized = torch.clip(resized, min=0.0, max=1.0)
            else:
                resized = torch.clip(resized, min=0, max=255)
            resized = resized.to(tensor.device).type(tensor.dtype)

            # Create padded tensor with background color
            # noinspection PyTypeChecker
            bg_color = torch.tensor(_parse_color_to_mode(
                self.background_color,
                mode={1: 'L', 2: 'LA', 3: 'RGB', 4: 'RGBA'}[c]
            ), device=tensor.device)
            if tensor.dtype.is_floating_point:
                bg_color = (bg_color / 255.0).type(tensor.dtype)
            else:
                bg_color = bg_color.type(tensor.dtype)

            result = bg_color.reshape(1, c, 1, 1).expand(b, c, target_h, target_w).clone()

            # Calculate padding positions
            pad_left = (target_w - new_w) // 2
            pad_top = (target_h - new_h) // 2

            # Paste resized image onto padded background
            result[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

            # Return to original batch dimension if needed
            if not is_batched:
                result = result.squeeze(0)

            return result

        def forward(self, pic):
            """
            Apply padding transform to input image.

            :param pic: Input image (PIL Image or torch.Tensor)
            :type pic: Union[PIL.Image.Image, torch.Tensor]

            :return: Padded image with target size
            :rtype: Union[PIL.Image.Image, torch.Tensor]
            :raises TypeError: If input is not a PIL Image or torch.Tensor
            """
            if isinstance(pic, Image.Image):
                return self._pad_pil_image(pic)
            elif isinstance(pic, torch.Tensor):
                return self._pad_tensor(pic)
            else:
                raise TypeError('pic should be PIL Image or a tensor. Got {}'.format(type(pic)))

        def __repr__(self) -> str:
            """
            Return string representation of the transform.

            :return: String representation
            :rtype: str
            """
            detail = f"(size={self.size}, interpolation={self.interpolation.value}, background_color={self.background_color})"
            return f"{self.__class__.__name__}{detail}"

else:
    PadToSize = None


@_register_transform('pad_to_size', safe=False)
def _create_pad_to_size(size: Union[Tuple[int, int], int],
                        background_color: Union[str, int, Tuple[int, int, int], Tuple[int, int, int, int]] = 'white',
                        interpolation='bilinear'):
    """
    Factory function to create PadToSize transform instance.

    :param size: Target size as (height, width) tuple or single int
    :type size: Union[Tuple[int, int], int]
    :param background_color: Color for padding
    :type background_color: Union[str, int, Tuple[int, int, int], Tuple[int, int, int, int]]
    :param interpolation: Interpolation mode name
    :type interpolation: str

    :return: PadToSize transform instance
    :rtype: PadToSize
    :raises AssertionError: If torchvision is not available
    """
    assert PadToSize is not None
    return PadToSize(
        size=size,
        background_color=background_color,
        interpolation=_get_interpolation_mode(interpolation),
    )


@_register_parse('pad_to_size', safe=False)
def _parse_pad_to_size(obj):
    """
    Parse PadToSize transform object for serialization.

    :param obj: Transform object to parse
    :type obj: Any

    :return: Dictionary containing transform parameters
    :rtype: dict
    :raises NotParseTarget: If object is not a PadToSize instance
    :raises AssertionError: If torchvision is not available
    """
    assert PadToSize is not None
    if not isinstance(obj, PadToSize):
        raise NotParseTarget

    obj: PadToSize
    return {
        'size': list(obj.size),
        'background_color': (list(obj.background_color)
                             if isinstance(obj.background_color, (list, tuple)) else obj.background_color),
        'interpolation': obj.interpolation.value,
    }


def create_torchvision_transforms(tvalue: Union[list, dict]):
    """
    Create torchvision transforms from config.

    :param tvalue: Transform configuration as list or dict
    :return: Composed transforms or single transform
    :raises TypeError: If tvalue has unsupported type

    :example:
        >>> from imgutils.preprocess import create_torchvision_transforms
        >>>
        >>> create_torchvision_transforms({
        ...     'type': 'resize',
        ...     'size': 384,
        ...     'interpolation': 'bicubic',
        ... })
        Resize(size=384, interpolation=bicubic, max_size=None, antialias=True)
        >>> create_torchvision_transforms({
        ...     'type': 'resize',
        ...     'size': (224, 256),
        ...     'interpolation': 'bilinear',
        ... })
        Resize(size=(224, 256), interpolation=bilinear, max_size=None, antialias=True)
        >>> create_torchvision_transforms({'type': 'center_crop', 'size': 224})
        CenterCrop(size=(224, 224))
        >>> create_torchvision_transforms({'type': 'to_tensor'})
        ToTensor()
        >>> create_torchvision_transforms({'type': 'maybe_to_tensor'})
        MaybeToTensor()
        >>> create_torchvision_transforms({'type': 'normalize', 'mean': 0.5, 'std': 0.5})
        Normalize(mean=0.5, std=0.5)
        >>> create_torchvision_transforms({
        ...     'type': 'normalize',
        ...     'mean': [0.485, 0.456, 0.406],
        ...     'std': [0.229, 0.224, 0.225],
        ... })
        Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
        >>> create_torchvision_transforms([
        ...     {'antialias': True,
        ...      'interpolation': 'bicubic',
        ...      'max_size': None,
        ...      'size': 384,
        ...      'type': 'resize'},
        ...     {'size': (224, 224), 'type': 'center_crop'},
        ...     {'type': 'maybe_to_tensor'},
        ...     {'mean': 0.5, 'std': 0.5, 'type': 'normalize'}
        ... ])
        Compose(
            Resize(size=384, interpolation=bicubic, max_size=None, antialias=True)
            CenterCrop(size=(224, 224))
            MaybeToTensor()
            Normalize(mean=0.5, std=0.5)
        )

    .. note::
        Currently the following transforms are supported:

        - `torchvision.transforms.Resize`
        - `torchvision.transforms.CenterCrop`
        - `torchvision.transforms.ToTensor`
        - `timm.data.MaybeToTensor`
        - `torchvision.transforms.Normalize`
    """
    _check_torchvision()

    from torchvision.transforms import Compose
    if isinstance(tvalue, list):
        return Compose([create_torchvision_transforms(titem) for titem in tvalue])
    elif isinstance(tvalue, dict):
        tvalue = copy.deepcopy(tvalue)
        ttype = tvalue.pop('type')
        return _TRANS_CREATORS[ttype](**tvalue)
    else:
        raise TypeError(f'Unknown type of transforms - {tvalue!r}.')


def parse_torchvision_transforms(value):
    """
    Parse torchvision transforms into config dict.

    :param value: Transform object to parse
    :return: Transform configuration as list or dict
    :raises TypeError: If transform type is not supported

    :example:
        >>> from timm.data import MaybeToTensor
        >>> from torchvision.transforms import Resize, InterpolationMode, CenterCrop, ToTensor, Normalize
        >>>
        >>> from imgutils.preprocess import parse_torchvision_transforms
        >>>
        >>> parse_torchvision_transforms(Resize(
        ...     size=384,
        ...     interpolation=InterpolationMode.BICUBIC,
        ... ))
        {'type': 'resize', 'size': 384, 'interpolation': 'bicubic', 'max_size': None, 'antialias': True}
        >>> parse_torchvision_transforms(Resize(
        ...     size=(224, 256),
        ...     interpolation=InterpolationMode.BILINEAR,
        ... ))
        {'type': 'resize', 'size': (224, 256), 'interpolation': 'bilinear', 'max_size': None, 'antialias': True}
        >>> parse_torchvision_transforms(CenterCrop(size=224))
        {'type': 'center_crop', 'size': (224, 224)}
        >>> parse_torchvision_transforms(ToTensor())
        {'type': 'to_tensor'}
        >>> parse_torchvision_transforms(MaybeToTensor())
        {'type': 'maybe_to_tensor'}
        >>> parse_torchvision_transforms(Normalize(mean=0.5, std=0.5))
        {'type': 'normalize', 'mean': 0.5, 'std': 0.5}
        >>> parse_torchvision_transforms(Normalize(
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225],
        ... ))
        {'type': 'normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        >>> parse_torchvision_transforms(Compose([
        ...     Resize(
        ...         size=384,
        ...         interpolation=Image.BICUBIC,
        ...     ),
        ...     CenterCrop(size=224),
        ...     MaybeToTensor(),
        ...     Normalize(mean=0.5, std=0.5),
        ... ]))
        [{'antialias': True,
          'interpolation': 'bicubic',
          'max_size': None,
          'size': 384,
          'type': 'resize'},
         {'size': (224, 224), 'type': 'center_crop'},
         {'type': 'maybe_to_tensor'},
         {'mean': 0.5, 'std': 0.5, 'type': 'normalize'}]
    """
    _check_torchvision()

    from torchvision.transforms import Compose
    if isinstance(value, Compose):
        return [
            parse_torchvision_transforms(trans)
            for trans in value.transforms
        ]
    else:
        for key, _parser in _TRANS_PARSERS.items():
            try:
                return _parser(value)
            except NotParseTarget:
                pass

        raise TypeError(f'Unknown parse transform - {value!r}.')
