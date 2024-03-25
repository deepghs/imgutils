import os.path

from imgutils.metrics import anime_dbaesthetic
from imgutils.metrics.dbaesthetic import _LABELS
from plot import image_plot

if __name__ == '__main__':
    items = []
    for label in _LABELS:
        image = os.path.join('dbaesthetic', f'{label}.jpg')
        l, p, s = anime_dbaesthetic(image, fmt=('label', 'percentile', 'score'))
        display_name = f'{l}\nScore: {s:.2f}/{6.0:.2f}\nPercentile: {p:.3f}'
        items.append((image, display_name))

    image_plot(
        *items,
        columns=4,
        figsize=(11, 9),
    )
