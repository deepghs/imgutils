import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

from zoo.camie.model.time import get_file_timestamp


class CamieTaggerInitial(nn.Module):
    """
    A lightweight version of ImageTagger that only includes the backbone and initial classifier.
    This model uses significantly less VRAM than the full model.
    """

    def __init__(self, total_tags: int, dropout: float = 0.1, pretrained: bool = True):
        super().__init__()
        # Core model config
        self.embedding_dim = 1280  # Fixed to EfficientNetV2-L output dimension

        # Initialize backbone
        weights = EfficientNet_V2_L_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_l(weights=weights)
        self.backbone.classifier = nn.Identity()

        # Spatial pooling only - no projection
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Initial tag prediction with bottleneck
        self.initial_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, total_tags)
        )

        # Temperature scaling
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def emb_to_pred(self, features):
        # Initial Tag Predictions
        initial_logits = self.initial_classifier(features)
        initial_preds = torch.clamp(initial_logits / self.temperature, min=-15.0, max=15.0)
        return initial_preds

    def forward(self, x):
        """Forward pass with only the initial predictions"""
        # Image Feature Extraction
        features = self.backbone.features(x)
        features = self.spatial_pool(features).squeeze(-1).squeeze(-1)

        # Initial Tag Predictions
        initial_preds = self.emb_to_pred(features)

        # For API compatibility with the full model, return the same predictions twice
        return features, initial_preds, features, initial_preds


class InitialEmbToPred(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model: CamieTaggerInitial = model

    def forward(self, features):
        return self.model.emb_to_pred(features)


def create_initial_model():
    repo_id = 'Camais03/camie-tagger'
    filename = 'model_initial.safetensors'
    safetensors_path = hf_hub_download(
        repo_id=repo_id,
        repo_type='model',
        filename=filename
    )
    state_dict = load_file(safetensors_path, device='cpu')
    # state_dict = torch.load(weights_path, map_location="cpu")
    # Instantiate the model with the same parameters as training
    model = CamieTaggerInitial(total_tags=70527, pretrained=True)  # dataset not needed for forward
    model.load_state_dict(state_dict)

    created_at = get_file_timestamp(
        repo_id=repo_id,
        repo_type='model',
        filename=filename
    )

    return model, created_at, (repo_id, filename), \
        (InitialEmbToPred(model), InitialEmbToPred(model))


if __name__ == '__main__':
    model, created_at, _, _ = create_initial_model()
    model.eval()  # set to evaluation mode
    print(model)

    # Define example input – a dummy image tensor of the expected input shape (1, 3, 512, 512)
    dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    with torch.no_grad():
        dummy_init_embeddings, dummy_init_logits, dummy_refined_embeddings, dummy_refined_logits = model(dummy_input)
    print(dummy_init_embeddings.shape, dummy_init_embeddings.dtype)
    print(dummy_init_logits.shape, dummy_init_logits.dtype)
    print(dummy_refined_embeddings.shape, dummy_refined_embeddings.dtype)
    print(dummy_refined_logits.shape, dummy_refined_logits.dtype)

    # # Export to ONNX
    # onnx_path = "camie_tagger_initial_v15.onnx"
    # torch.onnx.export(
    #     model, dummy_input, onnx_path,
    #     export_params=True,  # store the trained parameter weights in the model file
    #     opset_version=13,  # ONNX opset version (13 is widely supported)
    #     do_constant_folding=True,  # optimize constant expressions
    #     input_names=["input"],
    #     output_names=["initial_logits", "refined_logits"],
    #     # model.forward returns two outputs (identical for InitialOnly)
    #     dynamic_axes={"input": {0: "batch_size"}}  # allow variable batch size
    # )
    # print(f"ONNX model saved to: {onnx_path}")
