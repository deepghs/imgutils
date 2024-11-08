import numpy as np

__all__ = ['sigmoid']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
