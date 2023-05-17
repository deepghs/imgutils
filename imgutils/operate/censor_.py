from functools import lru_cache
from typing import Tuple, Type, List

from PIL import Image, ImageFilter

from imgutils.data import ImageTyping, load_image


class BaseCensor:
    def censor_area(self, image: Image.Image, area: Tuple[int, int, int, int], **kwargs) -> Image.Image:
        raise NotImplementedError  # pragma: no cover


class PixelateCensor(BaseCensor):
    def censor_area(self, image: Image.Image, area: Tuple[int, int, int, int], radius: int = 4,
                    **kwargs) -> Image.Image:
        image = image.copy()
        x0, y0, x1, y1 = area
        width, height = x1 - x0, y1 - y0
        censor_area = image.crop((x0, y0, x1, y1))
        censor_area = censor_area.resize((width // radius, height // radius)).resize((width, height), Image.NEAREST)
        image.paste(censor_area, (x0, y0, x1, y1))
        return image


class BlurCensor(BaseCensor):
    def censor_area(self, image: Image.Image, area: Tuple[int, int, int, int], radius: int = 4,
                    **kwargs) -> Image.Image:
        image = image.copy()
        x0, y0, x1, y1 = area
        censor_area = image.crop((x0, y0, x1, y1))
        censor_area = censor_area.filter(ImageFilter.GaussianBlur(radius))
        image.paste(censor_area, (x0, y0, x1, y1))
        return image


class ColorCensor(BaseCensor):
    def censor_area(self, image: Image.Image, area: Tuple[int, int, int, int], color: str = 'black',
                    **kwargs) -> Image.Image:
        image = image.copy()
        x0, y0, x1, y1 = area
        censor_area = image.crop((x0, y0, x1, y1))
        # noinspection PyTypeChecker
        censor_area = Image.new(image.mode, censor_area.size, color=color)
        image.paste(censor_area, (x0, y0, x1, y1))
        return image


_KNOWN_CENSORS = {}


def register_censor_method(name: str, cls: Type[BaseCensor], *args, **kwargs):
    if name in _KNOWN_CENSORS:
        raise KeyError(f'Censor method {name!r} already exist, please use another name.')
    _KNOWN_CENSORS[name] = (cls, args, kwargs)


register_censor_method('pixelate', PixelateCensor)
register_censor_method('blur', BlurCensor)
register_censor_method('color', ColorCensor)


@lru_cache()
def _get_censor_instance(name: str) -> BaseCensor:
    if name in _KNOWN_CENSORS:
        cls, args, kwargs = _KNOWN_CENSORS[name]
        return cls(*args, **kwargs)
    else:
        raise KeyError(f'Censor method {name!r} not found.')


def censor(image: ImageTyping, method: str, areas: List[Tuple[float, float, float, float]], **kwargs) -> Image.Image:
    image = load_image(image, mode='RGB')
    c = _get_censor_instance(method)
    for x0, y0, x1, y1 in areas:
        image = c.censor_area(image, (int(x0), int(y0), int(x1), int(y1)), **kwargs)

    return image
