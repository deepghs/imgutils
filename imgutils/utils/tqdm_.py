import io

from tqdm.auto import tqdm as _origin_tqdm

__all__ = ['tqdm']


def tqdm(*args, silent: bool = False, **kwargs):
    """
    An enhanced version of tqdm (progress bar) with an option to silence the output.

    This function modifies the behavior of tqdm to allow silencing the progress bar.

    :param args: Positional arguments to be passed to tqdm.
    :param silent: If True, the progress bar content will not be displayed.
    :type silent: bool
    :param kwargs: Additional keyword arguments to be passed to tqdm.
    :return: tqdm progress bar.
    :rtype: tqdm.std.tqdm
    """
    with io.StringIO() as sio:
        if silent:
            kwargs['file'] = sio

        return _origin_tqdm(*args, **kwargs)
