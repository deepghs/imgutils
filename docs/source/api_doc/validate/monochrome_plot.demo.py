import glob
import os.path

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *glob.glob(os.path.join('mono', '*.jpg')),
        *glob.glob(os.path.join('colored', '*.jpg')),
        save_as='monochrome.dat.svg',
        columns=3, figsize=(8, 12),
    )
