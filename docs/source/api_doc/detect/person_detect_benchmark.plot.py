from imgutils.detect import detect_person
from imgutils.detect.visual import detection_visualize
from plot import image_plot


def _detect(img, **kwargs):
    return detection_visualize(img, detect_person(img, **kwargs))


if __name__ == '__main__':
    image_plot(
        (_detect('nian.png'), 'large scale'),
        (_detect('two_bikini_girls.png'), 'closed faces'),
        (_detect('genshin_post.jpg'), 'multiple'),
        (_detect('soldiers.jpg'), 'multiple++'),
        columns=2,
        figsize=(12, 9),
    )
