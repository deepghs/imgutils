"""
Overview:
    Management of onnx models.
"""
import logging
import os
import shutil
from typing import Optional

from hbutils.system import pip_install

__all__ = [
    'get_onnx_provider', 'open_onnx_model'
]


def _ensure_onnxruntime():
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
    Overview:
        Get onnx provider.

    :param provider: The provider for ONNX runtime. ``None`` by default and will automatically detect
        if the ``CUDAExecutionProvider`` is available. If it is available, it will be used,
        otherwise the default ``CPUExecutionProvider`` will be used.
    :return: String of the provider.
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


def _open_onnx_model(ckpt: str, provider: str, use_cpu: bool = True) -> InferenceSession:
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "CPUExecutionProvider":
        options.intra_op_num_threads = os.cpu_count()

    providers = [provider]
    if use_cpu and "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    logging.info(f'Model {ckpt!r} loaded with provider {provider!r}')
    return InferenceSession(ckpt, options, providers=providers)


def open_onnx_model(ckpt: str, mode: str = None) -> InferenceSession:
    """
    Overview:
        Open an ONNX model and load its ONNX runtime.

    :param ckpt: ONNX model file.
    :param mode: Provider of the ONNX. Default is ``None`` which means the provider will be auto-detected,
        see :func:`get_onnx_provider` for more details.
    :return: A loaded ONNX runtime object.

    .. note::
        When ``mode`` is set to ``None``, it will attempt to detect the environment variable ``ONNX_MODE``.
        This means you can decide which ONNX runtime to use by setting the environment variable. For example,
        on Linux, executing ``export ONNX_MODE=cpu`` will ignore any existing CUDA and force the model inference
        to run on CPU.
    """
    return _open_onnx_model(ckpt, get_onnx_provider(mode or os.environ.get('ONNX_MODE', None)))
