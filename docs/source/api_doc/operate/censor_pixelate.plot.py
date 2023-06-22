from PIL import Image

from imgutils.operate import censor_areas
from plot import image_plot

if __name__ == '__main__':
    image = Image.open('genshin_post.jpg')
    areas = [
        (967, 143, 1084, 261),
        (246, 208, 331, 287),
        (662, 466, 705, 514),
        (479, 283, 523, 326)
    ]

    image_plot(
        (image, 'origin'),
        (censor_areas(image, 'pixelate', areas, radius=4), 'pixelate (radius=4)'),
        (censor_areas(image, 'pixelate', areas, radius=8), 'pixelate (radius=8)'),
        (censor_areas(image, 'pixelate', areas, radius=12), 'pixelate (radius=12)'),
        columns=2,
        figsize=(9.5, 6),
        autocensor=False,
    )
