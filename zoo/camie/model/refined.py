import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

from zoo.camie.model.time import get_file_timestamp


class MultiheadAttentionNoFlash(nn.Module):
    """Custom multi-head attention module (replaces FlashAttention) using ONNX-friendly ops."""

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # scaling factor for dot-product attention

        # Define separate projections for query, key, value, and output (no biases to match FlashAttention)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        # (Note: We omit dropout in attention computation for ONNX simplicity; model should be set to eval mode anyway.)

    def forward(self, query, key=None, value=None):
        # Allow usage as self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = key

        # Linear projections
        Q = self.q_proj(query)  # [B, S_q, dim]
        K = self.k_proj(key)  # [B, S_k, dim]
        V = self.v_proj(value)  # [B, S_v, dim]

        # Reshape into (B, num_heads, S, head_dim) for computing attention per head
        B, S_q, _ = Q.shape
        _, S_k, _ = K.shape
        Q = Q.view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, S_q, head_dim]
        K = K.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, S_k, head_dim]
        V = V.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, S_k, head_dim]

        # Scaled dot-product attention: compute attention weights
        attn_weights = torch.matmul(Q, K.transpose(2, 3))  # [B, heads, S_q, S_k]
        attn_weights = attn_weights * self.scale
        attn_probs = F.softmax(attn_weights, dim=-1)  # softmax over S_k (key length)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_probs, V)  # [B, heads, S_q, head_dim]

        # Reshape back to [B, S_q, dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S_q, self.dim)
        # Output projection
        output = self.out_proj(attn_output)  # [B, S_q, dim]
        return output


class CamieTaggerRefined(nn.Module):
    """
    Refined CAMIE Image Tagger model without FlashAttention.
    - EfficientNetV2 backbone
    - Initial classifier for preliminary tag logits
    - Multi-head self-attention on top predicted tag embeddings
    - Multi-head cross-attention between image feature and tag embeddings
    - Refined classifier for final tag logits
    """

    def __init__(self, total_tags: int, tag_context_size: int = 256, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.tag_context_size = tag_context_size
        self.embedding_dim = 1280  # EfficientNetV2-L feature dimension

        # Backbone feature extractor (EfficientNetV2-L)
        backbone = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
        backbone.classifier = nn.Identity()  # remove final classification head
        self.backbone = backbone

        # Spatial pooling to get a single feature vector per image (1x1 avg pool)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Initial classifier (two-layer MLP) to predict tags from image feature
        self.initial_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, total_tags)  # outputs raw logits for all tags
        )

        # Embedding for tags (each tag gets an embedding vector, used for attention)
        self.tag_embedding = nn.Embedding(total_tags, self.embedding_dim)

        # Self-attention over the selected tag embeddings (replaces FlashAttention)
        self.tag_attention = MultiheadAttentionNoFlash(self.embedding_dim, num_heads=num_heads, dropout=dropout)
        self.tag_norm = nn.LayerNorm(self.embedding_dim)

        # Projection from image feature to query vector for cross-attention
        self.cross_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        )
        # Cross-attention between image feature (as query) and tag features (as key/value)
        self.cross_attention = MultiheadAttentionNoFlash(self.embedding_dim, num_heads=num_heads, dropout=dropout)
        self.cross_norm = nn.LayerNorm(self.embedding_dim)

        # Refined classifier (takes concatenated original & attended features)
        self.refined_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, total_tags)
        )

        # Temperature parameter for scaling logits (to calibrate confidence)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, images):
        # 1. Feature extraction
        feats = self.backbone.features(images)  # [B, 1280, H/32, W/32] features
        feats = self.spatial_pool(feats).squeeze(-1).squeeze(-1)  # [B, 1280] global feature vector per image

        # 2. Initial tag prediction
        initial_logits = self.initial_classifier(feats)  # [B, total_tags]
        # Scale by temperature and clamp (to stabilize extreme values, as in original)
        initial_preds = torch.clamp(initial_logits / self.temperature, min=-15.0, max=15.0)

        # 3. Select top-k predicted tags for context (tag_context_size)
        probs = torch.sigmoid(initial_preds)  # convert logits to probabilities
        # Get indices of top `tag_context_size` tags for each sample
        _, topk_indices = torch.topk(probs, k=self.tag_context_size, dim=1)
        # 4. Embed selected tags
        tag_embeds = self.tag_embedding(topk_indices)  # [B, tag_context_size, embedding_dim]

        # 5. Self-attention on tag embeddings (to refine tag representation)
        attn_tags = self.tag_attention(tag_embeds)  # [B, tag_context_size, embedding_dim]
        attn_tags = self.tag_norm(attn_tags)  # layer norm

        # 6. Cross-attention between image feature and attended tags
        # Expand image features to have one per tag position
        feat_q = self.cross_proj(feats)  # [B, embedding_dim]
        # Repeat each image feature vector tag_context_size times to form a sequence
        feat_q = feat_q.unsqueeze(1).expand(-1, self.tag_context_size, -1)  # [B, tag_context_size, embedding_dim]
        # Use image features as queries, tag embeddings as keys and values
        cross_attn = self.cross_attention(feat_q, attn_tags, attn_tags)  # [B, tag_context_size, embedding_dim]
        cross_attn = self.cross_norm(cross_attn)

        # 7. Fuse features: average the cross-attended tag outputs, and combine with original features
        fused_feature = cross_attn.mean(dim=1)  # [B, embedding_dim]
        combined_feature = torch.cat([feats, fused_feature], dim=1)  # [B, embedding_dim*2]

        # 8. Refined tag prediction
        refined_logits = self.refined_classifier(combined_feature)  # [B, total_tags]
        refined_preds = torch.clamp(refined_logits / self.temperature, min=-15.0, max=15.0)

        return feats, initial_preds, combined_feature, refined_preds


def create_refined_model():
    repo_id = 'Camais03/camie-tagger'
    filename = 'model_refined.safetensors'

    # --- Load the pretrained refined model weights ---
    total_tags = 70527  # total number of tags in the dataset (Danbooru 2024)
    safetensors_path = hf_hub_download(
        repo_id=repo_id,
        repo_type='model',
        filename=filename,
    )
    state_dict = load_file(safetensors_path, device='cpu')  # Load the saved weights (should be an OrderedDict)
    # state_dict = torch.load("model_refined.pt", map_location="cpu")  # Load the saved weights (should be an OrderedDict)

    # Initialize our model and load weights
    model = CamieTaggerRefined(total_tags=total_tags)
    model.load_state_dict(state_dict)

    created_at = get_file_timestamp(
        repo_id=repo_id,
        repo_type='model',
        filename=filename,
    )

    return model, created_at, (repo_id, filename)


if __name__ == '__main__':
    model, created_at, _ = create_refined_model()
    model.eval()  # set to evaluation mode (disable dropout)
    print(model)

    # (Optional) Cast to float32 if weights were in half precision
    # model = model.float()

    # --- Export to ONNX ---
    dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    with torch.no_grad():
        dummy_init_embeddings, dummy_init_logits, dummy_refined_embeddings, dummy_refined_logits = model(dummy_input)
    print(dummy_init_embeddings.shape, dummy_init_embeddings.dtype)
    print(dummy_init_logits.shape, dummy_init_logits.dtype)
    print(dummy_refined_embeddings.shape, dummy_refined_embeddings.dtype)
    print(dummy_refined_logits.shape, dummy_refined_logits.dtype)

    # output_onnx_file = "camie_refined_no_flash_v15.onnx"
    # torch.onnx.export(
    #     model, dummy_input, output_onnx_file,
    #     export_params=True,  # store trained parameter weights inside the model file
    #     opset_version=17,  # ONNX opset version (ensure support for needed ops)
    #     do_constant_folding=True,  # optimize constant expressions
    #     input_names=["image"],
    #     output_names=["initial_tags", "refined_tags"],
    #     dynamic_axes={  # set batch dimension to be dynamic
    #         "image": {0: "batch"},
    #         "initial_tags": {0: "batch"},
    #         "refined_tags": {0: "batch"}
    #     }
    # )
    # print(f"ONNX model exported to {output_onnx_file}")
