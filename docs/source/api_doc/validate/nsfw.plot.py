import glob
import os.path

from natsort import natsorted

from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *natsorted(glob.glob(os.path.join('nsfw', 'drawings', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('nsfw', 'hentai', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('nsfw', 'neutral', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('nsfw', 'porn', '*.jpg'))),
        *natsorted(glob.glob(os.path.join('nsfw', 'sexy', '*.jpg'))),
        columns=4, figsize=(10, 18.5),
    )
