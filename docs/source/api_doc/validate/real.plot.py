import glob
import os.path

from natsort import natsorted

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *natsorted(glob.glob(os.path.join('real', 'anime', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('real', 'real', '*.jpg'))),
        columns=4, figsize=(10, 15),
    )
