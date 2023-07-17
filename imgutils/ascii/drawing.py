import io
import math
import shutil
from typing import Optional

import numpy as np
from PIL import Image

from ..data import ImageTyping, load_image

__all__ = [
    'ascii_drawing',
]


def ascii_drawing(img: ImageTyping, max_width: Optional[int] = ..., max_height: Optional[int] = ...,
                  resample=Image.BILINEAR, levels: str = "@%#*+=-:. "):
    if max_width is ... or max_height is ...:
        terminal = shutil.get_terminal_size(fallback=(80, 40))
        max_width = terminal.columns - 5 if max_width is ... else max_width
        max_height = terminal.lines - 5 if max_height is ... else max_height

    img = load_image(img, force_background='white', mode='RGB')
    if (max_width is not None and img.width > max_width) or \
            (max_height is not None and img.height > max_height):
        r1 = max_width / img.width if max_width else +math.inf
        r2 = max_height / img.height if max_height else +math.inf
        r = min(r1, r2)

        width = int(img.width * r)
        height = int(img.height * r)
        img = img.resize((width, height), resample=resample)
    img = img.convert('L')

    greyscale = np.array(img).astype(np.float32) / 255.0
    chids = np.clip((greyscale * len(levels)).astype(np.int32), a_min=0, a_max=len(levels) - 1)
    charr = np.array(list(levels))[chids]
    with io.StringIO() as sio:
        for row in charr:
            print(''.join(row.tolist()), file=sio)

        return sio.getvalue()
