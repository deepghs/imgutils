import copy
import io
from textwrap import indent
from typing import Union, Optional, Tuple, List

import numpy as np
from PIL import Image

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
    def _fn(func):
        _PTRANS_CREATORS[name] = func
        return func

    return _fn


class PillowResize:
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
            raise ValueError(
                "max_size is only supported for single int size or sequence of length 1"
            )

        # noinspection PyTypeChecker
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def _get_resize_size(self, img: Image.Image) -> Tuple[int, int]:
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
                if max(oh, ow) > max_size:
                    if oh > ow:
                        ow = int(max_size * ow / oh)
                        oh = max_size
                    else:
                        oh = int(max_size * oh / ow)
                        ow = max_size

            return ow, oh
        else:
            # noinspection PyUnresolvedReferences
            return self.size[1], self.size[0]

    def __call__(self, img: Image.Image) -> Image.Image:
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
        interpolate_str = _PILLOW_TO_STR[self.interpolation]
        detail = f"(size={self.size}, interpolation={interpolate_str}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


@register_pillow_transform('resize')
def _create_resize(size, interpolation='bilinear', max_size=None, antialias=True):
    return PillowResize(
        size=size,
        interpolation=_get_pillow_resample(interpolation),
        max_size=max_size,
        antialias=antialias,
    )


class PillowCenterCrop:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (tuple, list)) and len(size) == 1:
            self.size = (size[0], size[0])
        elif isinstance(size, (tuple, list)) and len(size) == 2:
            self.size = size
        else:
            raise ValueError("Please provide only two dimensions (h, w) for size.")

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image')

        return self._center_crop(img)

    def _center_crop(self, img):
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
        return f"{self.__class__.__name__}(size={self.size})"


@register_pillow_transform('center_crop')
def _create_center_crop(size):
    return PillowCenterCrop(
        size=size
    )


class PillowToTensor:
    def __init__(self):

        self.pil_modes = {'L', 'LA', 'P', 'I', 'F', 'RGB', 'YCbCr', 'RGBA', 'CMYK', '1'}

    def __call__(self, pic):
        if not isinstance(pic, Image.Image):
            raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

        if pic.mode == 'I':
            # 32-bit signed integer pixels
            return np.array(pic, np.int32, copy=True)
        elif pic.mode == 'I;16':
            # 16-bit signed integer pixels
            return np.array(pic, np.int16, copy=True)
        elif pic.mode == 'F':
            # 32-bit floating point pixels
            return np.array(pic, np.float32, copy=True)

        img = np.array(pic, copy=True)
        if pic.mode == '1':
            return img.astype(np.float32)
        elif pic.mode == 'L':
            img = img.reshape((1,) + img.shape)
            return img.astype(np.float32) / 255
        elif pic.mode == 'LA':
            img_l = img[..., 0].reshape((1,) + img.shape[:2])
            img_a = img[..., 1].reshape((1,) + img.shape[:2])
            return np.concatenate((img_l, img_a), axis=0).astype(np.float32) / 255
        elif pic.mode == 'P':
            pic = pic.convert('RGB')
            img = np.array(pic, copy=True)
            img = img.transpose((2, 0, 1))
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

        raise ValueError(f"Unsupported PIL image mode: {pic.mode}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@register_pillow_transform('to_tensor')
def _create_to_tensor():
    return PillowToTensor()


class PillowMaybeToTensor:
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            return image
        else:
            return PillowToTensor()(image)

    def __repr__(self):
        return f'{type(self).__name__}()'


@register_pillow_transform('maybe_to_tensor')
def _create_maybe_to_tensor():
    return PillowMaybeToTensor()


class PillowNormalize:
    def __init__(self, mean, std, inplace=False):
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
        if not isinstance(array, np.ndarray):
            raise TypeError('Input should be a numpy.ndarray')

        if array.dtype != np.float32:
            array = array.astype(np.float32)
        if not self.inplace:
            array = array.copy()

        if array.ndim == 2:
            if isinstance(self.mean, np.ndarray) or isinstance(self.std, np.ndarray):
                raise ValueError("Channel-wise mean/std can't be used for single channel data")
            return self._normalize_single(array)
        elif array.ndim >= 3:
            return self._normalize_multi(array)
        else:
            raise ValueError(f"Expected 2D or more dims array, got {array.ndim}D")

    def _normalize_single(self, array):
        mean = self.mean.reshape(1, 1)
        std = self.std.reshape(1, 1)
        array = (array - mean) / std
        return array

    def _normalize_multi(self, array):
        mean = self.mean.reshape(-1, 1, 1)
        std = self.std.reshape(-1, 1, 1)
        array = (array - mean) / std
        return array

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


@register_pillow_transform('normalize')
def _create_normalize(mean, std, inplace=False):
    return PillowNormalize(
        mean=mean,
        std=std,
        inplace=inplace,
    )


class PillowCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        x = image
        for trans in self.transforms:
            x = trans(x)
        return x

    def __repr__(self):
        with io.StringIO() as sf:
            print(f'{type(self).__name__}(', file=sf)
            for trans in self.transforms:
                print(indent(repr(trans), prefix='    '), file=sf)
            print(f')', file=sf)
            return sf.getvalue()


def create_pillow_transforms(tvalue: Union[list, dict]):
    if isinstance(tvalue, list):
        return PillowCompose([create_pillow_transforms(titem) for titem in tvalue])
    elif isinstance(tvalue, dict):
        tvalue = copy.deepcopy(tvalue)
        ttype = tvalue.pop('type')
        return _PTRANS_CREATORS[ttype](**tvalue)
    else:
        raise TypeError(f'Unknown type of transforms - {tvalue!r}.')
