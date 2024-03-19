"""
Overview:
    A utility for reading and writing metadata from/to model files in the A41 WebUI format.

    .. note::
        ``torch`` and ``safetensors`` are required by this model.
        Please install them with the following command before start using this module.

        .. code:: shell

            pip install dghs-imgutils[model]
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
    Checks if the required dependencies (Safetensors and Torch) are installed.
    Raises EnvironmentError if they are not installed.
    """
    if not safetensors:
        raise EnvironmentError(
            'Safetensors not installed. Please use "pip install dghs-imgutils[model]".')  # pragma: no cover
    if not torch:
        raise EnvironmentError(
            'Torch not installed. Please use "pip install dghs-imgutils[model]".')  # pragma: no cover


def read_metadata(model_file: str) -> Dict[str, str]:
    """
    Reads metadata from a model file and returns it as a dictionary.

    :param model_file: The path to the model file.
    :type model_file: str
    :return: The metadata extracted from the model file.
    :rtype: Dict[str, str]
    """
    _check_env()
    with safetensors.safe_open(model_file, 'pt') as f:
        return f.metadata()


def save_with_metadata(src_model_file: str, dst_model_file: str, metadata: Dict[str, str], clear: bool = False):
    """
    Saves a model file with metadata. Optionally, existing metadata can be cleared before adding new metadata.

    :param src_model_file: The path to the source model file.
    :type src_model_file: str
    :param dst_model_file: The path to save the new model file.
    :type dst_model_file: str
    :param metadata: The metadata to add to the model file.
    :type metadata: Dict[str, str]
    :param clear: Whether to clear existing metadata before adding new metadata. Default is False.
    :type clear: bool
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
