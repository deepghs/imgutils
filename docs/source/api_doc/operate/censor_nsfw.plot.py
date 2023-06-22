from PIL import Image

from imgutils.operate import censor_nsfw
from plot import image_plot

if __name__ == '__main__':
    image = Image.open('nude_girl.png')
    image_plot(
        (image, 'origin'),
        (censor_nsfw(image, 'color', nipple_f=True, color='black'), 'color (black)'),
        (censor_nsfw(image, 'pixelate', nipple_f=True), 'pixelate'),
        (censor_nsfw(image, 'emoji', nipple_f=True), 'emoji'),
        columns=2,
        figsize=(7, 8),
        autocensor=False,
    )
