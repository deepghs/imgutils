import glob
import os.path

from natsort import natsorted

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *natsorted(glob.glob(os.path.join('mono', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('colored', '*.jpg'))),
        columns=3, figsize=(8, 12),
    )
