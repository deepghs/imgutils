import font
from imgutils.data import load_image
from imgutils.detect import detect_text
from imgutils.detect.visual import detection_visualize
from imgutils.ocr import ocr
from plot import image_plot


def _detect_with_ocr(img, *, max_size=None, **kwargs):
    img = load_image(img, mode='RGB', force_background='white')
    if max_size is not None and min(img.height, img.width) > max_size:
        r = max_size / min(img.height, img.width)
        img = img.resize((
            int(round(img.width * r)),
            int(round(img.height * r)),
        ))

    return detection_visualize(img, ocr(img, **kwargs), fp=font.get_cn_fp())


def _detect_with_deprecated(img, **kwargs):
    return detection_visualize(img, detect_text(img, **kwargs))


if __name__ == '__main__':
    image_plot(
        (_detect_with_deprecated('text/ml2.jpg'), 'detect_text'),
        (_detect_with_ocr('text/ml2.jpg'), 'detect_text_with_ocr'),
        columns=2,
        figsize=(13, 3.8),
    )
