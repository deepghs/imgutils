import font
from imgutils.data import load_image
from imgutils.detect.visual import detection_visualize
from imgutils.ocr import ocr
from plot import image_plot

_max_size = 480


def _detect(img, **kwargs):
    img = load_image(img, mode='RGB', force_background='white')
    if min(img.height, img.width) > _max_size:
        r = _max_size / min(img.height, img.width)
        img = img.resize((
            int(round(img.width * r)),
            int(round(img.height * r)),
        ))

    return detection_visualize(img, ocr(img, **kwargs), fp=font.get_cn_fp())


if __name__ == '__main__':
    image_plot(
        (_detect('post_text.jpg', recognize_model='japan_PP-OCRv3_rec'), 'Text of Post'),
        (_detect('anime_subtitle.jpg'), 'Subtitle of Anime'),
        (_detect('comic.jpg'), 'Comic'),
        (_detect('plot.png'), 'Complex'),
        columns=2,
        figsize=(12, 9),
    )
