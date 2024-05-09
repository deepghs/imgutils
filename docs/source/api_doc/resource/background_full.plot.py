import random

from imgutils.resource import list_bg_image_files, get_bg_image
from plot import image_plot

if __name__ == '__main__':
    filenames = random.sample(list_bg_image_files(), 9)

    image_plot(
        *(
            (get_bg_image(filename), filename)
            for filename in filenames
        ),
        columns=3,
        figsize=(14, 10),
    )
