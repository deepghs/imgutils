import json
import os.path

import numpy as np
import onnx
import onnxruntime
import torch
from PIL import Image
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_fs, get_hf_client
from procslib import get_model
from procslib.models.pixai_tagger import PixAITaggerInference
from thop import profile, clever_format
from torch import nn

from imgutils.preprocess import parse_torchvision_transforms
from test.testings import get_testfile
from zoo.utils import onnx_optimize


class ModuleWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, classifier: nn.Module):
        super().__init__()
        self.base_module = base_module
        self.classifier = classifier

        self._output_features = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input_tensor, output_tensor):
            assert isinstance(input_tensor, tuple) and len(input_tensor) == 1
            input_tensor = input_tensor[0]
            self._output_features = input_tensor

        self.classifier.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor):
        logits = self.base_module(x)
        preds = torch.sigmoid(logits)

        if self._output_features is None:
            raise RuntimeError("Target module did not receive any input during forward pass")
        features, self._output_features = self._output_features, None
        assert all([x == 1 for x in features.shape[2:]]), f'Invalid feature shape: {features.shape!r}'
        features = torch.flatten(features, start_dim=1)

        return features, logits, preds


def load_model(model_name: str = "tagger_v_2_2_7"):
    model: PixAITaggerInference = get_model("pixai_tagger", model_version=model_name, device='cpu')
    infer_model = model.model
    transforms = model.transform
    return infer_model, transforms


def extract(export_dir: str, model_name: str = "tagger_v_2_2_7", no_optimize: bool = False):
    hf_fs = get_hf_fs()
    hf_client = get_hf_client()

    os.makedirs(export_dir, exist_ok=True)

    model, transforms = load_model(model_name)
    image = Image.open(get_testfile('genshin_post.jpg'))
    dummy_input = transforms(image).unsqueeze(0)
    logging.info(f'Dummy input size: {dummy_input.shape!r}')

    with torch.no_grad():
        expected_dummy_output = model(dummy_input)
    logging.info(f'Dummy output size: {expected_dummy_output.shape!r}')

    classifier = model.get_classifier()
    classifier_position = None
    for name, module in model.named_modules():
        if module is classifier:
            classifier_position = name
            break
    if not classifier_position:
        raise RuntimeError(f'No classifier module found in model {type(model)}.')
    logging.info(f'Classifier module found at {classifier_position!r}:\n{classifier}')

    wrapped_model = ModuleWrapper(model, classifier=classifier)
    with torch.no_grad():
        conv_features, conv_output, conv_preds = wrapped_model(dummy_input)
    logging.info(f'Shape of embeddings: {conv_features.shape!r}')
    logging.info(f'Sample of expected logits:\n'
                 f'{expected_dummy_output[:, -10:]}\n'
                 f'Sample of actual logits:\n'
                 f'{conv_output[:, -10:]}')
    close_matrix = torch.isclose(expected_dummy_output, conv_output, atol=1e-3)
    ratio = close_matrix.type(torch.float32).mean()
    logging.info(f'{ratio * 100:.2f}% of the logits value are the same.')
    assert close_matrix.all(), 'Not all values can match.'

    logging.info('Profiling model ...')
    macs, params = profile(model, inputs=(dummy_input,))
    s_macs, s_params = clever_format([macs, params], "%.1f")
    logging.info(f'Params: {s_params}, FLOPs: {s_macs}')

    with open(os.path.join(export_dir, 'preprocess.json'), 'w') as f:
        json.dump({
            'stages': parse_torchvision_transforms(transforms),
        }, f, indent=4, sort_keys=True)

    onnx_filename = os.path.join(export_dir, 'model.onnx')
    with TemporaryDirectory() as td:
        temp_model_onnx = os.path.join(td, 'model.onnx')
        logging.info(f'Exporting temporary ONNX model to {temp_model_onnx!r} ...')
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            temp_model_onnx,
            input_names=['input'],
            output_names=['embedding', 'logits', 'output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'embedding': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            },
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            custom_opsets=None,
        )

        model = onnx.load(temp_model_onnx)
        if not no_optimize:
            logging.info('Optimizing onnx model ...')
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        logging.info(f'Complete model saving to {onnx_filename!r} ...')
        onnx.save(model, onnx_filename)

        session = onnxruntime.InferenceSession(onnx_filename)
        o_logits, o_embeddings = session.run(['logits', 'embedding'], {'input': dummy_input.numpy()})
        emb_1 = o_embeddings / np.linalg.norm(o_embeddings, axis=-1, keepdims=True)
        emb_2 = conv_features.numpy() / np.linalg.norm(conv_features.numpy(), axis=-1, keepdims=True)
        emb_sims = (emb_1 * emb_2).sum()
        logging.info(f'Similarity of the embeddings is {emb_sims:.5f}.')
        assert emb_sims >= 0.98, f'Similarity of the embeddings is {emb_sims:.5f}, ONNX validation failed.'


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    extract(
        export_dir='test_ex',
    )
