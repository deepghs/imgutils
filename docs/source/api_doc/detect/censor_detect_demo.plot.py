from imgutils.detect import detect_censors
from imgutils.detect.censor import _LABELS
from imgutils.detect.visual import detection_visualize
from plot import image_plot


def _detect(img, **kwargs):
    return detection_visualize(img, detect_censors(img, **kwargs), _LABELS)


if __name__ == '__main__':
    image_plot(
        (_detect('censor/nude_girl.png'), 'simple nude'),
        (_detect('censor/simple_sex.jpg'), 'simple sex'),
        (_detect('censor/complex_pose.jpg'), 'complex pose'),
        (_detect('censor/complex_sex.jpg'), 'complex sex'),
        columns=2,
        figsize=(9, 9),
        autocensor=False,
    )