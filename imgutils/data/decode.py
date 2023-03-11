import numpy as np
from PIL import Image

__all__ = [
    'rgb_decode'
]

_DEFAULT_ORDER = 'HWC'


def _get_hwc_map(order_: str):
    order_ = order_.upper()
    return tuple(order_.index(c) for c in _DEFAULT_ORDER.upper())


_float_types = [np.float16, np.float32, np.float64]
if hasattr(np, 'float128'):
    _float_types.append(np.float128)
_float_types = tuple(_float_types)


def rgb_decode(data, order_: str = 'CHW') -> Image.Image:
    """
    Overview:
        Decode numpy data to ``PIL.Image.Image``.

    :param data: Original numpy data (both ``np.uint8`` and ``np.float32`` are supported).
    :param order_: Order of the given ``data``.
    :return: Decoded pil image object.

    Examples::
        >>> from PIL import Image
        >>> from imgutils.data import rgb_encode, rgb_decode
        >>>
        >>> image = Image.open('custom_image.jpg')
        >>> data = rgb_encode(image)
        >>> data_cwh = rgb_encode(image, order_='CWH')
        >>> data_int = rgb_encode(image, use_float=False)
        >>>
        >>> rgb_decode(data)
        <PIL.Image.Image image mode=RGB size=1606x1870 at 0x7FB9B89BBDC0>
        >>> rgb_decode(data_cwh, order_='CWH')
        <PIL.Image.Image image mode=RGB size=1606x1870 at 0x7FB9B89BBE50>
        >>> rgb_decode(data_int)
        <PIL.Image.Image image mode=RGB size=1606x1870 at 0x7FB9B89BBDF0>

    .. note::
        :func:`rgb_decode` is the inverse operation of :func:`imgutils.data.encode.rgb_encode`.
    """
    if data.dtype in (np.uint8, np.int8, np.uint16, np.int16,
                      np.uint32, np.int32, np.uint64, np.int64):
        data = data.astype(np.uint8)
    elif data.dtype in _float_types:
        data = np.clip(data, 0.0, 1.0)
        data = (data * 255).astype(np.uint8)
    else:
        raise TypeError(f'Unknown dtype for data - {data.dtype!r}.')  # pragma: no cover

    data = np.transpose(data, _get_hwc_map(order_))
    return Image.fromarray(data, 'RGB')
