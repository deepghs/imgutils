import glob
import os.path

from natsort import natsorted

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *natsorted(glob.glob(os.path.join('aicheck', 'ai', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('aicheck', 'human', '*.jpg'))),
        columns=3, figsize=(8, 12),
    )
