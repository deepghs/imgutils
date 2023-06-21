from imgutils.detect import detect_faces
from imgutils.detect.visual import detection_visualize
from plot import image_plot


def _detect(img, **kwargs):
    return detection_visualize(img, detect_faces(img, **kwargs))


if __name__ == '__main__':
    image_plot(
        (_detect('censor/nude_girl.png'), 'simple nude'),
        (_detect('censor/simple_sex.jpg'), 'simple sex'),
        (_detect('censor/complex_pose.jpg'), 'complex pose'),
        (_detect('censor/complex_sex.jpg'), 'complex sex'),
        columns=2,
        figsize=(12, 9),
        autocensor=False,
    )