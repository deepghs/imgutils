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
        (censor_areas(image, 'emoji', areas), 'emoji'),
        (censor_areas(image, 'emoji', areas, emoji=':cat_face:'), 'emoji (cat_face)'),
        (censor_areas(image, 'emoji', areas, emoji='😅'), 'emoji\n(grinning_face_with_sweat)'),
        columns=2,
        figsize=(9.5, 6),
    )
