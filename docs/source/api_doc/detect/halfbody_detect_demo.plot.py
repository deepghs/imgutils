from imgutils.detect import detect_halfbodies
from imgutils.detect.halfbody import _LABELS
from imgutils.detect.visual import detection_visualize
from plot import image_plot


def _detect(img, **kwargs):
    return detection_visualize(img, detect_halfbodies(img, **kwargs), _LABELS)


if __name__ == '__main__':
    image_plot(
        (_detect('halfbody/upper.png'), 'upper body'),
        (_detect('halfbody/full.png'), 'more body'),
        (_detect('halfbody/lie.jpg'), 'lie'),
        (_detect('halfbody/squat.jpg'), 'squat'),
        columns=2,
        figsize=(9, 9),
        autocensor=False,
    )
