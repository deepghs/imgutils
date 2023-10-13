import math

import numpy as np

from .tqdm_ import tqdm

__all__ = ['area_batch_run']


def area_batch_run(origin_input: np.ndarray, func, scale: int = 1,
                   tile_size: int = 512, tile_overlap: int = 16, batch_size: int = 4,
                   input_channels: int = 3, output_channels: int = 3, silent: bool = False,
                   process_title: str = 'Process Tiles', rebuild_title: str = 'Rebuild Tiles'):
    """
    Perform a batch execution of a given function on overlapping tiles of a large image.

    This function divides the original input image into tiles, applies a given function to each tile,
    and then reconstructs the image by combining the processed tiles.

    :param origin_input: The original input image as a NumPy ndarray with shape (batch, channels, height, width).
    :type origin_input: np.ndarray
    :param func: The function to apply to each tile. It should accept a tile (np.ndarray)
        as input and return the processed tile.
    :type func: callable
    :param scale: Scaling factor for output, defaults to 1.
    :type scale: int, optional
    :param tile_size: Size of the tiles, defaults to 512.
    :type tile_size: int, optional
    :param tile_overlap: Overlap between adjacent tiles, defaults to 16.
    :type tile_overlap: int, optional
    :param batch_size: Batch size for processing tiles, defaults to 4.
    :type batch_size: int, optional
    :param input_channels: Number of input channels, defaults to 3.
    :type input_channels: int, optional
    :param output_channels: Number of output channels, defaults to 3.
    :type output_channels: int, optional
    :param silent: If True, suppresses the progress bar output, defaults to False.
    :type silent: bool, optional
    :param process_title: Title for the processing progress bar, defaults to 'Process Tiles'.
    :type process_title: str, optional
    :param rebuild_title: Title for the rebuilding progress bar, defaults to 'Rebuild Tiles'.
    :type rebuild_title: str, optional
    :return: Processed image as a NumPy ndarray.
    :rtype: np.ndarray
    """
    batch, channels, height, width = origin_input.shape
    assert channels == input_channels, f'Input channels {input_channels!r} expected, but {channels!r} found.'

    tile = min(tile_size, height, width)
    stride = tile - tile_overlap
    h_idx_list = sorted(set(list(range(0, height - tile, stride)) + [height - tile]))
    w_idx_list = sorted(set(list(range(0, width - tile, stride)) + [width - tile]))
    sum_ = np.zeros((batch, output_channels, height * scale, width * scale), dtype=origin_input.dtype)
    weight = np.zeros_like(sum_, dtype=origin_input.dtype)

    all_patch = []
    all_idx = []

    with tqdm(total=math.ceil(len(h_idx_list) * len(w_idx_list) / batch_size),
              desc=process_title, silent=silent) as pbar:
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = origin_input[..., h_idx: h_idx + tile, w_idx: w_idx + tile]
                all_patch.append(in_patch)
                all_idx.append((h_idx, w_idx))

        results = []
        for i in range(0, len(all_patch), batch_size):
            input_ = np.concatenate(all_patch[i:i + batch_size])
            output_ = func(input_)
            for idx, (h_idx, w_idx) in enumerate(all_idx[i:i + batch_size]):
                results.append((h_idx, w_idx, output_[idx]))
            pbar.update()

    for h_idx, w_idx, output_ in tqdm(results, desc=rebuild_title, silent=silent):
        out_patch_mask = np.ones_like(output_)
        h_min, h_max = h_idx * scale, (h_idx + tile) * scale
        w_min, w_max = w_idx * scale, (w_idx + tile) * scale
        sum_[..., h_min:h_max, w_min:w_max] += output_
        weight[..., h_min:h_max, w_min:w_max] += out_patch_mask

    return sum_ / weight
