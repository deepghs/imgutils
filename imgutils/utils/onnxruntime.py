"""
Overview:
    Management of ONNX models with automatic runtime detection and provider selection.

    This module provides utilities for loading and managing ONNX models with support for
    different execution providers (CPU, CUDA, TensorRT). It automatically handles the
    installation of onnxruntime based on the system configuration and provides a
    convenient interface for model inference.
"""
import logging
import os
import shutil
import warnings
from typing import Optional

from hbutils.system import pip_install

__all__ = [
    'get_onnx_provider', 'open_onnx_model'
]


def _ensure_onnxruntime():
    """
    Ensure that onnxruntime is installed on the system.

    This function automatically detects if NVIDIA GPU is available and installs
    the appropriate version of onnxruntime (GPU or CPU version).

    :raises ImportError: If installation fails
    """
    try:
        import onnxruntime
    except (ImportError, ModuleNotFoundError):
        logging.warning('Onnx runtime not installed, preparing to install ...')
        if shutil.which('nvidia-smi'):
            logging.info('Installing onnxruntime-gpu ...')
            pip_install(['onnxruntime-gpu'], silent=True)
        else:
            logging.info('Installing onnxruntime (cpu) ...')
            pip_install(['onnxruntime'], silent=True)


_ensure_onnxruntime()
from onnxruntime import get_available_providers, get_all_providers, InferenceSession, SessionOptions, \
    GraphOptimizationLevel

alias = {
    'gpu': "CUDAExecutionProvider",
    "trt": "TensorrtExecutionProvider",
}


def get_onnx_provider(provider: Optional[str] = None):
    """
    Get the appropriate ONNX execution provider based on system capabilities and user preference.

    This function automatically detects available execution providers and returns the most
    suitable one. It supports aliases for common providers and falls back to CPU execution
    if GPU providers are not available.

    :param provider: The provider for ONNX runtime. ``None`` by default and will automatically detect
        if the ``CUDAExecutionProvider`` is available. If it is available, it will be used,
        otherwise the default ``CPUExecutionProvider`` will be used. Supported aliases include
        'gpu' for CUDAExecutionProvider and 'trt' for TensorrtExecutionProvider.
    :type provider: Optional[str]

    :return: String name of the selected execution provider.
    :rtype: str

    :raises ValueError: If the specified provider is not supported or available.

    Example::
        >>> # Auto-detect provider
        >>> provider = get_onnx_provider()
        >>> print(provider)  # 'CUDAExecutionProvider' or 'CPUExecutionProvider'

        >>> # Explicitly request GPU provider
        >>> provider = get_onnx_provider('gpu')
        >>> print(provider)  # 'CUDAExecutionProvider'

        >>> # Request CPU provider
        >>> provider = get_onnx_provider('cpu')
        >>> print(provider)  # 'CPUExecutionProvider'
    """
    if not provider:
        if "CUDAExecutionProvider" in get_available_providers():
            return "CUDAExecutionProvider"
        else:
            return "CPUExecutionProvider"
    elif provider.lower() in alias:
        return alias[provider.lower()]
    else:
        for p in get_all_providers():
            if provider.lower() == p.lower() or f'{provider}ExecutionProvider'.lower() == p.lower():
                return p

        raise ValueError(f'One of the {get_all_providers()!r} expected, '
                         f'but unsupported provider {provider!r} found.')


def _open_onnx_model(ckpt: str, provider: str, use_cpu: bool = True,
                     cuda_device_id: Optional[int] = None) -> InferenceSession:
    """
    Internal function to create and configure an ONNX inference session.

    This function handles the low-level configuration of the ONNX runtime session,
    including optimization settings and provider-specific configurations.

    :param ckpt: Path to the ONNX model file.
    :type ckpt: str
    :param provider: Name of the execution provider to use.
    :type provider: str
    :param use_cpu: Whether to include CPU provider as fallback. Defaults to True.
    :type use_cpu: bool
    :param cuda_device_id: Specific CUDA device ID to use for GPU inference.
    :type cuda_device_id: Optional[int]

    :return: Configured ONNX inference session.
    :rtype: InferenceSession
    """
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "CPUExecutionProvider":
        options.intra_op_num_threads = os.cpu_count()

    if provider == 'CUDAExecutionProvider' and cuda_device_id is not None:
        providers = [
            ('CUDAExecutionProvider', {'device_id': cuda_device_id}),
        ]
    else:
        if provider != 'CUDAExecutionProvider' and cuda_device_id is not None:
            warnings.warn(UserWarning(
                'CUDA device ID specified but provider is not CUDAExecutionProvider. The device ID will be ignored.'))
        providers = [provider]
    if use_cpu and "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    logging.info(f'Model {ckpt!r} loaded with provider {provider!r}')
    return InferenceSession(ckpt, options, providers=providers)


def open_onnx_model(ckpt: str, mode: str = None, cuda_device_id: Optional[int] = None) -> InferenceSession:
    """
    Open an ONNX model and create a configured inference session.

    This function provides a high-level interface for loading ONNX models with
    automatic provider selection and optimization. It supports environment variable
    configuration for runtime provider selection.

    :param ckpt: Path to the ONNX model file to load.
    :type ckpt: str
    :param mode: Provider of the ONNX runtime. Default is ``None`` which means the provider will be auto-detected,
        see :func:`get_onnx_provider` for more details. Can also be controlled via ONNX_MODE environment variable.
    :type mode: Optional[str]
    :param cuda_device_id: Specific CUDA device ID to use for GPU inference. Only effective when using CUDA provider.
    :type cuda_device_id: Optional[int]

    :return: A loaded and configured ONNX inference session ready for prediction.
    :rtype: InferenceSession

    .. note::
        When ``mode`` is set to ``None``, it will attempt to detect the environment variable ``ONNX_MODE``.
        This means you can decide which ONNX runtime to use by setting the environment variable. For example,
        on Linux, executing ``export ONNX_MODE=cpu`` will ignore any existing CUDA and force the model inference
        to run on CPU.

    Example::
        >>> # Load model with auto-detected provider
        >>> session = open_onnx_model('model.onnx')

        >>> # Force CPU execution
        >>> session = open_onnx_model('model.onnx', mode='cpu')

        >>> # Use specific CUDA device
        >>> session = open_onnx_model('model.onnx', mode='gpu', cuda_device_id=1)
    """
    return _open_onnx_model(
        ckpt=ckpt,
        provider=get_onnx_provider(mode or os.environ.get('ONNX_MODE', None)),
        use_cpu=True,
        cuda_device_id=cuda_device_id,
    )
