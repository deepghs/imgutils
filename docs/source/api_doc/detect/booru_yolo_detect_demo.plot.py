from imgutils.detect.booru_yolo import detect_with_booru_yolo, _get_booru_yolo_labels, _DEFAULT_MODEL
from imgutils.detect.visual import detection_visualize
from plot import image_plot


def _detect(img, **kwargs):
    return detection_visualize(img, detect_with_booru_yolo(img, **kwargs), _get_booru_yolo_labels(_DEFAULT_MODEL))


if __name__ == '__main__':
    image_plot(
        (_detect('booru_yolo/nude_girl.png'), 'simple nude'),
        (_detect('booru_yolo/simple_sex.jpg'), 'simple sex'),
        (_detect('booru_yolo/two_bikini_girls.png'), '2 girls'),
        (_detect('booru_yolo/complex_sex.jpg'), 'complex sex'),
        columns=2,
        figsize=(9, 9),
        autocensor=False,
    )
