import copy
from functools import wraps
from typing import Union

from .base import NotParseTarget


def _check_torchvision():
    try:
        import torchvision
    except (ImportError, ModuleNotFoundError):
        raise EnvironmentError('No torchvision available.\n'
                               'Please install it by `pip install dghs-imgutils[torchvision]`.')


def _get_interpolation_mode(value):
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
    if safe:
        _check_torchvision()

    def _fn(func):
        _TRANS_CREATORS[name] = func
        return func

    return _fn


def register_torchvision_transform(name: str):
    return _register_transform(name, safe=True)


_TRANS_PARSERS = {}


def _register_parse(name: str, safe: bool = True):
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
    return _register_parse(name, safe=True)


@_register_transform('resize', safe=False)
def _create_resize(size, interpolation='bilinear', max_size=None, antialias=True):
    from torchvision.transforms import Resize
    return Resize(
        size=size,
        interpolation=_get_interpolation_mode(interpolation),
        max_size=max_size,
        antialias=antialias,
    )


@_register_parse('resize', safe=False)
def _parse_resize(obj):
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
    from torchvision.transforms import CenterCrop
    return CenterCrop(
        size=size,
    )


@_register_parse('center_crop', safe=False)
def _parse_center_crop(obj):
    from torchvision.transforms import CenterCrop
    if not isinstance(obj, CenterCrop):
        raise NotParseTarget

    obj: CenterCrop
    return {
        'size': obj.size,
    }


@_register_transform('maybe_to_tensor', safe=False)
def _create_maybe_to_tensor():
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

    return MaybeToTensor()


@_register_parse('maybe_to_tensor', safe=False)
def _parse_maybe_to_tensor(obj):
    if type(obj).__name__ != 'MaybeToTensor':
        raise NotParseTarget

    return {}


@_register_transform('to_tensor', safe=False)
def _create_to_tensor():
    from torchvision.transforms import ToTensor
    return ToTensor()


@_register_parse('to_tensor', safe=False)
def _parse_to_tensor(obj):
    if type(obj).__name__ != 'ToTensor':
        raise NotParseTarget

    return {}


@_register_transform('normalize', safe=False)
def _create_normalize(mean, std, inplace=False):
    import torch
    from torchvision.transforms import Normalize
    return Normalize(
        mean=torch.tensor(mean),
        std=torch.tensor(std),
        inplace=inplace,
    )


@_register_parse('normalize', safe=False)
def _parse_normalize(obj):
    from torchvision.transforms import Normalize
    if not isinstance(obj, Normalize):
        raise NotParseTarget

    obj: Normalize
    return {
        'mean': obj.mean.tolist(),
        'std': obj.std.tolist(),
    }


def create_torchvision_transforms(tvalue: Union[list, dict]):
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
