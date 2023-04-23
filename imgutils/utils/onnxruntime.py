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


def _open_onnx_model(ckpt: str, provider: str) -> InferenceSession:
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "CPUExecutionProvider":
        options.intra_op_num_threads = os.cpu_count()

    logging.info(f'Model {ckpt!r} loaded with provider {provider!r}')
    return InferenceSession(ckpt, options, [provider])


def open_onnx_model(ckpt: str, mode: str = None) -> InferenceSession:
    return _open_onnx_model(ckpt, get_onnx_provider(mode or os.environ.get('ONNX_MODE', None)))
