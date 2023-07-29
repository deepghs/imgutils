from imgutils.detect import detect_hands
from imgutils.detect.visual import detection_visualize
from plot import image_plot


def _detect(img, **kwargs):
    return detection_visualize(img, detect_hands(img, **kwargs))


if __name__ == '__main__':
    image_plot(
        (_detect('two_bikini_girls.png'), 'closed heads'),
        (_detect('mostima_post.jpg'), 'anime style'),
        columns=2,
        figsize=(12, 9),
    )
