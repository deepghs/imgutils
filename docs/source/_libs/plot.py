from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cli import _wrap_func_as_cli
from imgutils.data import load_image, grid_transparent, ImageTyping
from imgutils.detect import detect_censors
from imgutils.operate.censor_ import censor_areas
from imgutils.validate.truncate import _mock_load_truncated_images

INCHES_TO_PIXELS = 96


def _image_input_process(img, autocensor: bool = True) -> Tuple[Image.Image, str]:
    if isinstance(img, tuple):
        img_file, label = img
        image = load_image(img_file, force_background=None)
    elif isinstance(img, str):
        label = img
        image = load_image(img, force_background=None)
    else:
        raise TypeError(f'Unknown type of img - {img!r}.')

    image = grid_transparent(image)
    label = label.rstrip()

    if autocensor:
        detection = detect_censors(image)
        if detection:
            image = censor_areas(image, method='emoji', areas=[area for area, _, _ in detection])
            label = f'{label}\n(Censored)'

    return image, label


@_wrap_func_as_cli
@_mock_load_truncated_images(True)
def image_plot(*images, save_as: str, columns=2, keep_axis: bool = False, figsize=(6, 6), autocensor: bool = True):
    plt.cla()
    plt.tight_layout()

    assert images, 'No less than 1 images required.'
    n = len(images)
    rows = (n + columns - 1) // columns
    fig, axs = plt.subplots(rows, columns, figsize=figsize)
    if rows == 1 and columns == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = axs[None, ...]
    elif columns == 1:
        axs = axs[..., None]
    else:
        pass
    plt.subplots_adjust(wspace=0.2, hspace=0.15)
    for i, img in enumerate(images, start=0):
        xi, yi = i // columns, i % columns
        image, label = _image_input_process(img, autocensor)
        if rows == 1 and columns == 1:
            ax = axs
        elif rows == 1:
            ax = axs[yi]
        else:
            ax = axs[xi, yi]
        ax.imshow(image)
        ax.set_title(label)
        if not keep_axis:
            ax.axis('off')

    for i in range(len(images), rows * columns):
        xi, yi = i // columns, i % columns
        ax = axs[yi] if rows == 1 else axs[xi, yi]
        ax.axis('off')

    plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=300, transparent=True)


@_wrap_func_as_cli
@_mock_load_truncated_images(True)
def image_table(images: List[List[ImageTyping]], columns: List[str], rows: List[str], save_as: str,
                keep_axis: bool = False, figsize=(720, 600), dpi: int = 300, fontsize: int = 18, padsize: int = 5):
    plt.cla()
    plt.tight_layout()

    assert images, 'No less than 1 images required.'
    fig, axs = plt.subplots(len(rows), len(columns),
                            figsize=(figsize[0] / INCHES_TO_PIXELS, figsize[1] / INCHES_TO_PIXELS))
    plt.subplots_adjust(wspace=padsize / INCHES_TO_PIXELS, hspace=padsize / INCHES_TO_PIXELS)
    for xi in range(len(rows)):
        for yi in range(len(columns)):
            img = images[xi][yi]
            ax = axs[yi] if len(rows) == 1 else axs[xi, yi]
            if img:
                image = load_image(img, force_background=None)
                ax.imshow(grid_transparent(image))
                if not keep_axis:
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            if xi == 0:
                ax.set_title(columns[yi], fontsize=fontsize)
            if yi == 0:
                ax.set_ylabel(rows[xi], fontsize=fontsize)

    plt.savefig(save_as, bbox_inches='tight', pad_inches=0.02, dpi=dpi, transparent=True)
