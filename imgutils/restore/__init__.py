"""
Overview:
    Utilities for restoring images, which may be jpeg, blurry or noisy.

    The following models are used:

    * `NafNet <https://github.com/megvii-research/NAFNet>`_
    * `SCUNet <https://github.com/cszn/SCUNet>`_

    .. image:: restore_demo.plot.py.svg
        :align: center

"""
from .adversarial import remove_adversarial_noise
from .nafnet import restore_with_nafnet
from .scunet import restore_with_scunet
