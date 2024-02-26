import glob
import os.path

from natsort import natsorted

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *natsorted(glob.glob(os.path.join('classify', '3d', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('classify', 'bangumi', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('classify', 'comic', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('classify', 'illustration', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('classify', 'not_painting', '*.jpg'))),
        columns=3, figsize=(8, 15),
    )
