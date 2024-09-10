"""
A utility module for reading and writing metadata from/to model files in the A41 WebUI format.

This module provides functions to read metadata from model files and save model files with custom metadata.
It supports operations on SafeTensors files, which are a safe and efficient way to store tensors.

.. note::

    ``torch`` and ``safetensors`` are required by this module.
    Please install them with the following command before using this module:

    .. code:: shell

        pip install dghs-imgutils[model]

Functions:

    - read_metadata: Reads metadata from a model file.
    - save_with_metadata: Saves a model file with custom metadata.
    - _check_env: Internal function to check if required dependencies are installed.

Dependencies:

    - torch
    - safetensors
"""

from typing import Dict

try:
    import torch
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    torch = None

try:
    import safetensors.torch
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    safetensors = None


def _check_env():
    """
    Check if the required dependencies (Safetensors and Torch) are installed.

    This function verifies that both Safetensors and Torch are available in the current
    environment. If either is missing, it raises an EnvironmentError with instructions
    on how to install the missing dependency.

    :raises EnvironmentError: If Safetensors or Torch is not installed.
    """
    if not safetensors:
        raise EnvironmentError(
            'Safetensors not installed. Please use "pip install dghs-imgutils[model]".')  # pragma: no cover
    if not torch:
        raise EnvironmentError(
            'Torch not installed. Please use "pip install dghs-imgutils[model]".')  # pragma: no cover


def read_metadata(model_file: str) -> Dict[str, str]:
    """
    Read metadata from a model file and return it as a dictionary.

    This function opens the specified model file using SafeTensors and extracts
    its metadata. The metadata is returned as a dictionary where both keys and
    values are strings.

    :param model_file: The path to the model file to read metadata from.
    :type model_file: str
    :return: A dictionary containing the metadata of the model file.
    :rtype: Dict[str, str]
    :raises EnvironmentError: If required dependencies are not installed.

    Usage:
        >>> metadata = read_metadata("path/to/model.safetensors")
        >>> print(metadata)
        {'key1': 'value1', 'key2': 'value2', ...}
    """
    _check_env()
    with safetensors.safe_open(model_file, 'pt') as f:
        return f.metadata()


def save_with_metadata(src_model_file: str, dst_model_file: str, metadata: Dict[str, str], clear: bool = False):
    """
    Save a model file with custom metadata.

    This function reads the source model file, adds or updates its metadata,
    and saves it to a new destination file. It can optionally clear existing
    metadata before adding new metadata.

    :param src_model_file: The path to the source model file.
    :type src_model_file: str
    :param dst_model_file: The path where the new model file will be saved.
    :type dst_model_file: str
    :param metadata: A dictionary of metadata to add or update in the model file.
    :type metadata: Dict[str, str]
    :param clear: If True, clear existing metadata before adding new metadata. Default is False.
    :type clear: bool
    :raises EnvironmentError: If required dependencies are not installed.

    Usage:
        >>> new_metadata = {"author": "John Doe", "version": "1.0"}
        >>> save_with_metadata("input_model.safetensors", "output_model.safetensors", new_metadata)
        # This will add or update the "author" and "version" metadata in the new file

        >>> save_with_metadata("input_model.safetensors", "output_model.safetensors", new_metadata, clear=True)
        # This will clear all existing metadata and only include the new metadata in the output file
    """
    _check_env()
    with safetensors.safe_open(src_model_file, framework='pt') as f:
        if clear:
            new_metadata = {**(metadata or {})}
        else:
            new_metadata = {**f.metadata(), **(metadata or {})}
        safetensors.torch.save_file(
            tensors={key: f.get_tensor(key) for key in f.keys()},
            filename=dst_model_file,
            metadata=new_metadata,
        )
