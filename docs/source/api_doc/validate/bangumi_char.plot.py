import glob
import os.path

from natsort import natsorted

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *natsorted(glob.glob(os.path.join('bangumi_char', 'vision', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('bangumi_char', 'imagery', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('bangumi_char', 'halfbody', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('bangumi_char', 'face', '*.jpg'))),
        columns=4, figsize=(10, 15),
    )
