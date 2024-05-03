import os.path

from natsort import natsorted

from imgutils.metrics import laplacian_score
from plot import image_plot

if __name__ == '__main__':
    items = []
    for filename in natsorted(os.listdir('laplacian')):
        image = os.path.join('laplacian', filename)
        ls = laplacian_score(image)
        display_name = f'{filename}\nScore: {ls:.3f}'
        items.append((image, display_name))

    image_plot(
        *items,
        columns=2,
        figsize=(6, 5),
    )
