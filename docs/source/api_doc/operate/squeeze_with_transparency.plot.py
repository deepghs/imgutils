from PIL import Image

from imgutils.operate import squeeze_with_transparency
from plot import image_plot

if __name__ == '__main__':
    image = Image.open('jerry_with_space.png')

    image_plot(
        (image, 'origin'),
        (squeeze_with_transparency(image), 'squeezed'),
        columns=2,
        figsize=(7, 4),
        keep_axis=True,
    )
