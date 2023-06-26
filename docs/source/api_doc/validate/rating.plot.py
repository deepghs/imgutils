import glob
import os.path

from natsort import natsorted

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *natsorted(glob.glob(os.path.join('rating', 'safe', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('rating', 'r15', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('rating', 'r18', '*.jpg'))),
        columns=4, figsize=(10, 12),
    )
