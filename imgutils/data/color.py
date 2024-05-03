import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans

from .image import load_image, ImageTyping
from ..utils import vreplace

__all__ = [
    'extract_main_colors',
]


def _image_resize(image: Image.Image, pixels: int = 90000, **kwargs):
    """
    Resize the input image.

    :param image: The input image to resize.
    :type image: Image.Image
    :param pixels: The maximum number of pixels. (default: 90000)
    :type pixels: int
    :param kwargs: Additional keyword arguments for resizing.
    :return: The resized image.
    :rtype: Image.Image
    """
    rt = (image.size[0] * image.size[1] / pixels) ** 0.5
    if rt > 1.0:
        small_image = image.resize((int(image.size[0] / rt), int(image.size[1] / rt)), **kwargs)
    else:
        small_image = image.copy()
    return small_image


def extract_main_colors(image: ImageTyping, n: int = 12, scale: int = 400, mode: str = 'LAB',
                        force_background: str = 'white', fmt=('image', 'colors'), **kwargs):
    """
    Extracts the main colors from the input image using KMeans clustering.

    :param image: The input image.
    :type image: ImageTyping
    :param n: The number of main colors to extract. (default: 12)
    :type n: int
    :param scale: The scale factor for resizing the image. (default: 400)
    :type scale: int
    :param mode: The mode of the input image. (default: 'LAB')
    :type mode: str
    :param force_background: The background color to force. (default: 'white')
    :type force_background: str
    :param fmt: The format of the output. (default: ('image', 'colors'))
    :type fmt: tuple
    :param kwargs: Additional keyword arguments for image loading and resizing.
    :return: The extracted main colors and the resized image.
    :rtype: tuple
    """
    image = load_image(image, mode='RGB', force_background=force_background)
    image = image.convert(mode)
    bands = image.getbands()
    small_image = _image_resize(image, scale ** 2, **kwargs)

    few_raw = np.asarray(small_image).reshape(-1, len(bands))
    kmeans = KMeans(n_clusters=n, n_init='auto')
    kmeans.fit(few_raw)

    width, height = image.size
    raw = np.asarray(image).reshape(-1, len(bands))
    colors = kmeans.cluster_centers_.round().astype(np.uint8)
    prediction = kmeans.predict(raw)
    new_data = colors[prediction].reshape((height, width, len(bands)))
    new_image = Image.fromarray(new_data, mode=mode)

    cids, counts = np.unique(prediction, return_counts=True)
    counts = np.asarray(list(map(lambda x: x[1], sorted(zip(cids, counts)))))
    total_counts = counts.sum()

    records = []
    for i, (cc, count) in enumerate(zip(colors, counts)):
        records.append({
            'i': i,
            **{band: c for band, c in zip(bands, cc)},
            'ratio': float(count / total_counts)
        })
    df_records = pd.DataFrame(records)
    df_records = df_records.sort_values('ratio', ascending=False)

    p = prediction.reshape((height, width))
    mx = np.array([v for _, v in sorted(zip(df_records['i'], range(len(df_records))))])
    del df_records['i']
    df_records['i'] = range(len(df_records))
    df_records = df_records[['i', *bands, 'ratio']]

    return vreplace(fmt, {
        'image': new_image,
        'colors': df_records,
        'prediction': mx[p],
    })
