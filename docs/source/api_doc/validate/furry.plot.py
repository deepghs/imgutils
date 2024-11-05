import glob
import os.path

from natsort import natsorted

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *natsorted(glob.glob(os.path.join('furry', 'non_furry', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('furry', 'furry', '*.jpg'))),
        columns=4, figsize=(10, 8),
    )
