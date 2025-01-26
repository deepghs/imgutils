"""
This module provides utilities for creating and parsing torchvision transforms.
It includes functionality for registering custom transforms, handling interpolation modes,
and converting between different transform representations.

The module supports common image transformations like resize, center crop, tensor conversion
and normalization. It provides a flexible framework for extending with additional transforms.
"""

import copy
from functools import wraps
from typing import Union

from .base import NotParseTarget

try:
    import torchvision
except (ImportError, ModuleNotFoundError):
    _HAS_TORCHVISION = False
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
