import torch
from torch import nn


class InitialOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        init_embeddings, init_logits, _, _ = self.model(x)
        init_prediction = torch.sigmoid(init_logits)
        return init_embeddings, init_logits, init_prediction


class FullWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        init_embeddings, init_logits, refined_embeddings, refined_logits = self.model(x)
        init_prediction = torch.sigmoid(init_logits)
        refined_prediction = torch.sigmoid(refined_logits)
        return init_embeddings, init_logits, init_prediction, \
            refined_embeddings, refined_logits, refined_prediction


class EmbToPredWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        logits = self.model(features)
        prediction = torch.sigmoid(logits)
        return logits, prediction
