import copy
from typing import Union


def _check_torchvision():
    try:
        import torchvision
    except (ImportError, ModuleNotFoundError):
        raise EnvironmentError('No torchvision available.\n'
                               'Please install it by `pip install dghs-imgutils[torchvision]`.')


def _get_interpolation_mode(value: Union[int, str]):
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


def _register(name: str, safe: bool = True):
    if safe:
        _check_torchvision()

    def _fn(func):
        _TRANS_CREATORS[name] = func
        return func

    return _fn


def register_torchvision_transform(name: str):
    _register(name, safe=True)


@_register('resize', safe=False)
def _create_resize(size, interpolation='bilinear', max_size=None, antialias=True):
    from torchvision.transforms import Resize
    return Resize(
        size=size,
        interpolation=_get_interpolation_mode(interpolation),
        max_size=max_size,
        antialias=antialias,
    )


@_register('center_crop', safe=False)
def _create_center_crop(size):
    from torchvision.transforms import CenterCrop
    return CenterCrop(
        size=size,
    )


@_register('maybe_to_tensor', safe=False)
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


@_register('to_tensor', safe=False)
def _create_to_tensor():
    from torchvision.transforms import ToTensor
    return ToTensor()


@_register('normalize', safe=False)
def _create_normalize(mean, std, inplace=False):
    import torch
    from torchvision.transforms import Normalize
    return Normalize(
        mean=torch.tensor(mean),
        std=torch.tensor(std),
        inplace=inplace,
    )


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
