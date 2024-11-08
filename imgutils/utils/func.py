"""
This module provides mathematical functions related to neural networks.

It includes the sigmoid activation function, which is commonly used in various
machine learning and deep learning models. The sigmoid function maps any input
value to a value between 0 and 1, making it useful for binary classification
problems and as an activation function in neural network layers.

Usage:
    >>> from imgutils.utils import sigmoid
    >>> result = sigmoid(input_value)
"""

import numpy as np

__all__ = ['sigmoid']


def sigmoid(x):
    """
    Compute the sigmoid function for the input.

    The sigmoid function is defined as:
    :math:`f\\left(x\\right) = \\frac{1}{1 + e^{-x}}`

    This function applies the sigmoid activation to either a single number
    or an array of numbers using NumPy for efficient computation.

    :param x: Input value or array of values.
    :type x: float or numpy.ndarray

    :return: Sigmoid of the input.
    :rtype: float or numpy.ndarray

    :example:
        >>> import numpy as np
        >>> sigmoid(0)
        0.5
        >>> sigmoid(np.array([-1, 0, 1]))
        array([0.26894142, 0.5       , 0.73105858])
    """
    return 1 / (1 + np.exp(-x))
