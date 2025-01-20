from typing import Union, Sequence, Optional, Tuple

from PIL import Image

_INT_TO_PILLOW = {
    0: Image.NEAREST,
    2: Image.BILINEAR,
    3: Image.BICUBIC,
    4: Image.BOX,
    5: Image.HAMMING,
    1: Image.LANCZOS
}

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


class PillowResize:
    def __init__(
            self,
            size: Union[int, Sequence[int]],
            interpolation: int = Image.BILINEAR,
            max_size: Optional[int] = None,
            antialias: bool = True
    ):
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")

        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def _get_resize_size(self, img: Image.Image) -> Tuple[int, int]:
        w, h = img.size

        if isinstance(self.size, int) or (isinstance(self.size, Sequence) and len(self.size) == 1):
            size = self.size if isinstance(self.size, int) else self.size[0]
            if (w <= h and w == size) or (h <= w and h == size):
                return w, h

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            if self.max_size is not None:
                if isinstance(self.size, int) or len(self.size) == 1:
                    max_size = self.max_size
                    if max(oh, ow) > max_size:
                        if oh > ow:
                            ow = int(max_size * ow / oh)
                            oh = max_size
                        else:
                            oh = int(max_size * oh / ow)
                            ow = max_size
                else:
                    raise ValueError(
                        "max_size is only supported for single int size or sequence of length 1"
                    )

            return ow, oh
        else:
            return self.size[1], self.size[0]

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError('Input must be a PIL Image')

        size = self._get_resize_size(img)
        width, height = size
        if width != img.width or height != img.height:
            if self.interpolation in {Image.BILINEAR, Image.BICUBIC}:
                return img.resize(size, self.interpolation, reducing_gap=None if self.antialias else 1.0)
            else:
                return img.resize(size, self.interpolation)

    def __repr__(self) -> str:
        interpolate_str = _PILLOW_TO_STR[self.interpolation]
        detail = f"(size={self.size}, interpolation={interpolate_str}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"
