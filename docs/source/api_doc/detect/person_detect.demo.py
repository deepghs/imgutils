from imgutils.detect import detect_person
from imgutils.detect.visual import detection_visualize
from plot import image_plot


def _detect(img, **kwargs):
    return detection_visualize(img, detect_person(img, **kwargs))


if __name__ == '__main__':
    image_plot(
        (_detect('genshin_post.jpg'), ''),
        save_as='person_detect.dat.svg',
        columns=1,
        figsize=(12, 9),
    )
