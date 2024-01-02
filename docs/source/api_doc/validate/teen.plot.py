import glob
import os.path

from natsort import natsorted

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *natsorted(glob.glob(os.path.join('teen', 'contentious', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('teen', 'safe_teen', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('teen', 'non_teen', '*.jpg'))),
        columns=3, figsize=(8, 10),
    )
