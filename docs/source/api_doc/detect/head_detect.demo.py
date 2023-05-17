from imgutils.detect import detect_heads
from imgutils.detect.visual import detection_visualize
from plot import image_plot


def _detect(img, **kwargs):
    return detection_visualize(img, detect_heads(img, **kwargs))


if __name__ == '__main__':
    image_plot(
        (_detect('nian.png'), 'large scale'),
        (_detect('two_bikini_girls.png'), 'closed heads'),
        (_detect('genshin_post.jpg'), 'multiple'),
        (_detect('mostima_post.jpg'), 'anime style'),
        save_as='head_detect.dat.svg',
        columns=2,
        figsize=(12, 9),
    )
