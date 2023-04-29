from typing import Tuple

import matplotlib.pyplot as plt
from PIL import Image

from imgutils.data import load_image, grid_transparent
from imgutils.validate.truncate import _mock_load_truncated_images


def _image_input_process(img) -> Tuple[Image.Image, str]:
    if isinstance(img, tuple):
        img_file, label = img
        image = load_image(img_file, force_background=None)
    elif isinstance(img, str):
        label = img
        image = load_image(img, force_background=None)
    else:
        raise TypeError(f'Unknown type of img - {img!r}.')

    return grid_transparent(image), label


@_mock_load_truncated_images(True)
def image_plot(*images, save_as: str, columns=2, keep_axis: bool = False, figsize=(6, 6)):
    plt.cla()
    plt.tight_layout()

    assert images, 'No less than 1 images required.'
    n = len(images)
    rows = (n + columns - 1) // columns
    fig, axs = plt.subplots(rows, columns, figsize=figsize)
    plt.subplots_adjust(wspace=0.2, hspace=0.15)
    for i, img in enumerate(images, start=0):
        xi, yi = i // columns, i % columns
        image, label = _image_input_process(img)
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
