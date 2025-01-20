import copy
from typing import Union

try:
    import torchvision
except (ImportError, ModuleNotFoundError):
    raise EnvironmentError('No torchvision available.\n'
                           'Please install it by `pip install torchvision`.')

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, Resize, Compose, CenterCrop, ToTensor, Normalize

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


def _get_interpolation_mode(value: Union[int, str, InterpolationMode]):
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


def register_torchvision_transform(name: str):
    def _fn(func):
        _TRANS_CREATORS[name] = func
        return func

    return _fn


@register_torchvision_transform('resize')
def _create_resize(size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True):
    return Resize(
        size=size,
        interpolation=_get_interpolation_mode(interpolation),
        max_size=max_size,
        antialias=antialias,
    )


@register_torchvision_transform('center_crop')
def _create_center_crop(size):
    return CenterCrop(
        size=size,
    )


class MaybeToTensor(ToTensor):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        return F.to_tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@register_torchvision_transform('maybe_to_tensor')
def _create_maybe_to_tensor():
    return MaybeToTensor()


@register_torchvision_transform('normalize')
def _create_normalize(mean, std, inplace=False):
    return Normalize(
        mean=torch.tensor(mean),
        std=torch.tensor(std),
        inplace=inplace,
    )


def create_torchvision_transforms(tvalue: Union[list, dict]):
    if isinstance(tvalue, list):
        return Compose([create_torchvision_transforms(titem) for titem in tvalue])
    elif isinstance(tvalue, dict):
        tvalue = copy.deepcopy(tvalue)
        ttype = tvalue.pop('type')
        return _TRANS_CREATORS[ttype](**tvalue)
    else:
        raise TypeError(f'Unknown type of transforms - {tvalue!r}.')
