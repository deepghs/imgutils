import logging

import torch
from torch import nn


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
        preds = self.base_module(x)

        if self._output_features is None:
            raise RuntimeError("Target module did not receive any input during forward pass")
        features, self._output_features = self._output_features, None
        assert all([x == 1 for x in features.shape[2:]]), f'Invalid feature shape: {features.shape!r}'
        features = torch.flatten(features, start_dim=1)

        return features, preds


def get_model(model: nn.Module, dummy_input: torch.Tensor):
    assert isinstance(model, nn.Sequential)
    head = model[-1]
    wrapped_model = ModuleWrapper(model, head)

    logging.info(f'Input size: {dummy_input.shape!r}')
    with torch.no_grad():
        dummy_embedding, dummy_preds = wrapped_model(dummy_input)
    logging.info(f'Embedding size: {dummy_embedding.shape!r}')
    logging.info(f'Preds size: {dummy_preds.shape!r}')

    return wrapped_model, (dummy_embedding, dummy_preds)
    # print(model[-1])
