from plot import image_plot

from imgutils.detect import detect_text
from imgutils.detect.visual import detection_visualize


def _detect(img, **kwargs):
    return detection_visualize(img, detect_text(img, **kwargs))


if __name__ == '__main__':
    image_plot(
        (_detect('text/ml1.png'), 'Multiple Languages I'),
        (_detect('text/ml2.jpg'), 'Multiple Languages II'),
        columns=1,
        figsize=(8, 9),
    )
