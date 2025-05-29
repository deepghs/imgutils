import glob

from natsort import natsorted

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *natsorted(glob.glob('classify_timm/*.jpg')),
        columns=3,
        figsize=(6.7, 7.5),
    )
