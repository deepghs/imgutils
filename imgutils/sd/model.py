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
    if not safetensors:
        raise EnvironmentError(
            'Safetensors not installed. Please use "pip install dghs-imgutils[model]".')  # pragma: no cover
    if not torch:
        raise EnvironmentError(
            'Torch not installed. Please use "pip install dghs-imgutils[model]".')  # pragma: no cover


def read_metadata(model_file: str) -> Dict[str, str]:
    _check_env()
    with safetensors.safe_open(model_file, 'pt') as f:
        return f.metadata()


def save_with_metadata(src_model_file: str, dst_model_file: str, metadata: Dict[str, str], clear: bool = False):
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
