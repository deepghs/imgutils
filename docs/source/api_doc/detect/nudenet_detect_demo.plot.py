from imgutils.detect.nudenet import _LABELS, detect_with_nudenet
from imgutils.detect.visual import detection_visualize
from plot import image_plot


def _detect(img, **kwargs):
    return detection_visualize(img, detect_with_nudenet(img, **kwargs), _LABELS)


if __name__ == '__main__':
    image_plot(
        (_detect('nudenet/nude_girl.png'), 'simple nude'),
        (_detect('nudenet/simple_sex.jpg'), 'simple sex'),
        (_detect('nudenet/complex_pose.jpg'), 'complex pose'),
        (_detect('nudenet/complex_sex.jpg'), 'complex sex'),
        columns=2,
        figsize=(9, 9),
        autonudenet=False,
    )
