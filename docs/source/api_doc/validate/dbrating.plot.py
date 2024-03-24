import glob
import os.path

from natsort import natsorted

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *natsorted(glob.glob(os.path.join('dbrating', 'general', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('dbrating', 'sensitive', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('dbrating', 'questionable', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('dbrating', 'explicit', '*.jpg'))),
        columns=4, figsize=(10, 15),
    )
