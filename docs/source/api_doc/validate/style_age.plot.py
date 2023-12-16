import glob
import os.path

from natsort import natsorted

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *natsorted(glob.glob(os.path.join('style_age', '1970s-', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('style_age', '1980s', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('style_age', '1990s', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('style_age', '2000s', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('style_age', '2010s', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('style_age', '2015s', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('style_age', '2020s', '*.jpg'))),
        columns=4, figsize=(10, 26),
    )
