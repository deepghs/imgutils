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
        (censor_areas(image, 'blur', areas, radius=4), 'blur (radius=4)'),
        (censor_areas(image, 'blur', areas, radius=8), 'blur (radius=8)'),
        (censor_areas(image, 'blur', areas, radius=12), 'blur (radius=12)'),
        columns=2,
        figsize=(9.5, 6),
    )
