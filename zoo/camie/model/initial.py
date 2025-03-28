import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights


class InitialOnlyImageTagger(nn.Module):
    """
    A lightweight version of ImageTagger that only includes the backbone and initial classifier.
    This model uses significantly less VRAM than the full model.
    """

    def __init__(self, total_tags, dataset, model_name='efficientnet_v2_l',
                 dropout=0.1, pretrained=True):
        super().__init__()
        # Debug and stats flags
        self._flags = {
            'debug': False,
            'model_stats': False
        }

        # Core model config
        self.dataset = dataset
        self.embedding_dim = 1280  # Fixed to EfficientNetV2-L output dimension

        # Initialize backbone
        if model_name == 'efficientnet_v2_l':
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

    @property
    def debug(self):
        return self._flags['debug']

    @debug.setter
    def debug(self, value):
        self._flags['debug'] = value

    @property
    def model_stats(self):
        return self._flags['model_stats']

    @model_stats.setter
    def model_stats(self, value):
        self._flags['model_stats'] = value

    def preprocess_image(self, image_path, image_size=512):
        """Process an image for inference using same preprocessing as training"""
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found at path: {image_path}")

        # Initialize the same transform used during training
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        try:
            with Image.open(image_path) as img:
                # Convert RGBA or Palette images to RGB
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                # Get original dimensions
                width, height = img.size
                aspect_ratio = width / height

                # Calculate new dimensions to maintain aspect ratio
                if aspect_ratio > 1:
                    new_width = image_size
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = image_size
                    new_width = int(new_height * aspect_ratio)

                # Resize with LANCZOS filter
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Create new image with padding
                new_image = Image.new('RGB', (image_size, image_size), (0, 0, 0))
                paste_x = (image_size - new_width) // 2
                paste_y = (image_size - new_height) // 2
                new_image.paste(img, (paste_x, paste_y))

                # Apply transforms (without normalization)
                img_tensor = transform(new_image)
                return img_tensor
        except Exception as e:
            raise Exception(f"Error processing {image_path}: {str(e)}")

    def forward(self, x):
        """Forward pass with only the initial predictions"""
        # Image Feature Extraction
        features = self.backbone.features(x)
        features = self.spatial_pool(features).squeeze(-1).squeeze(-1)

        # Initial Tag Predictions
        initial_logits = self.initial_classifier(features)
        initial_preds = torch.clamp(initial_logits / self.temperature, min=-15.0, max=15.0)

        # For API compatibility with the full model, return the same predictions twice
        return initial_preds, initial_preds

    def predict(self, image_path, threshold=0.325, category_thresholds=None):
        """
        Run inference on an image with support for category-specific thresholds.
        """
        # Preprocess the image
        img_tensor = self.preprocess_image(image_path).unsqueeze(0)

        # Move to the same device as model and convert to half precision
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype  # Match model's precision
        img_tensor = img_tensor.to(device, dtype=dtype)

        # Run inference
        with torch.no_grad():
            initial_preds, _ = self.forward(img_tensor)

            # Apply sigmoid to get probabilities
            initial_probs = torch.sigmoid(initial_preds)

            # Apply thresholds
            if category_thresholds:
                # Create binary prediction tensors
                initial_binary = torch.zeros_like(initial_probs)

                # Apply thresholds by category
                for category, cat_threshold in category_thresholds.items():
                    # Create a mask for tags in this category
                    category_mask = torch.zeros_like(initial_probs, dtype=torch.bool)

                    # Find indices for this category
                    for tag_idx in range(initial_probs.size(-1)):
                        try:
                            _, tag_category = self.dataset.get_tag_info(tag_idx)
                            if tag_category == category:
                                category_mask[:, tag_idx] = True
                        except:
                            continue

                    # Apply threshold only to tags in this category
                    cat_threshold_tensor = torch.tensor(cat_threshold, device=device, dtype=dtype)
                    initial_binary[category_mask] = (initial_probs[category_mask] >= cat_threshold_tensor).to(dtype)

                predictions = initial_binary
            else:
                # Use the same threshold for all tags
                threshold_tensor = torch.tensor(threshold, device=device, dtype=dtype)
                predictions = (initial_probs >= threshold_tensor).to(dtype)

            # Return the same probabilities for both initial and refined for API compatibility
            return {
                'initial_probabilities': initial_probs,
                'refined_probabilities': initial_probs,  # Same as initial for compatibility
                'predictions': predictions
            }

    def get_tags_from_predictions(self, predictions, include_probabilities=True):
        """
        Convert model predictions to human-readable tags grouped by category.
        """
        # Get non-zero predictions
        if predictions.dim() > 1:
            predictions = predictions[0]  # Remove batch dimension

        # Get indices of positive predictions
        indices = torch.where(predictions > 0)[0].cpu().tolist()

        # Group by category
        result = {}
        for idx in indices:
            tag_name, category = self.dataset.get_tag_info(idx)

            if category not in result:
                result[category] = []

            if include_probabilities:
                prob = predictions[idx].item()
                result[category].append((tag_name, prob))
            else:
                result[category].append(tag_name)

        # Sort tags by probability within each category
        if include_probabilities:
            for category in result:
                result[category] = sorted(result[category], key=lambda x: x[1], reverse=True)

        return result


class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, batch_first=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=0.1)

        self.scale = self.head_dim ** -0.5
        self.debug = False

    def _debug_print(self, name, tensor):
        """Debug helper"""
        if self.debug:
            print(f"\n{name}:")
            print(f"Shape: {tensor.shape}")
            print(f"Device: {tensor.device}")
            print(f"Dtype: {tensor.dtype}")
            if tensor.is_floating_point():
                with torch.no_grad():
                    print(f"Range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
                    print(f"Mean: {tensor.mean().item():.3f}")
                    print(f"Std: {tensor.std().item():.3f}")

    def _reshape_for_flash(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input tensor for flash attention format"""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # [B, H, S, D]
        return x.contiguous()

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with flash attention"""
        if self.debug:
            print("\nFlashAttention Forward Pass")

        batch_size = query.size(0)

        # Use query as key/value if not provided
        key = query if key is None else key
        value = query if value is None else value

        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.debug:
            self._debug_print("Query before reshape", q)

        # Reshape for attention [B, H, S, D]
        q = self._reshape_for_flash(q)
        k = self._reshape_for_flash(k)
        v = self._reshape_for_flash(v)

        if self.debug:
            self._debug_print("Query after reshape", q)

        # Handle masking
        if mask is not None:
            # First convert mask to proper shape based on input dimensionality
            if mask.dim() == 2:  # [B, S]
                mask = mask.view(batch_size, 1, -1, 1)
            elif mask.dim() == 3:  # [B, S, S]
                mask = mask.view(batch_size, 1, mask.size(1), mask.size(2))
            elif mask.dim() == 5:  # [B, 1, S, S, S]
                mask = mask.squeeze(1).view(batch_size, 1, mask.size(2), mask.size(3))

            # Ensure mask is float16 if we're using float16
            mask = mask.to(q.dtype)

            if self.debug:
                self._debug_print("Prepared mask", mask)
                print(f"q shape: {q.shape}, mask shape: {mask.shape}")

            # Create attention mask that covers the full sequence length
            seq_len = q.size(2)
            if mask.size(-1) != seq_len:
                # Pad or trim mask to match sequence length
                new_mask = torch.zeros(batch_size, 1, seq_len, seq_len,
                                       device=mask.device, dtype=mask.dtype)
                min_len = min(seq_len, mask.size(-1))
                new_mask[..., :min_len, :min_len] = mask[..., :min_len, :min_len]
                mask = new_mask

            # Create key padding mask
            key_padding_mask = mask.squeeze(1).sum(-1) > 0
            key_padding_mask = key_padding_mask.view(batch_size, 1, -1, 1)

            # Apply the key padding mask
            k = k * key_padding_mask
            v = v * key_padding_mask

        if self.debug:
            self._debug_print("Query before attention", q)
            self._debug_print("Key before attention", k)
            self._debug_print("Value before attention", v)

        # Run flash attention
        dropout_p = self.dropout if self.training else 0.0
        output = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=self.scale,
            causal=False
        )

        if self.debug:
            self._debug_print("Output after attention", output)

        # Reshape output [B, H, S, D] -> [B, S, H, D] -> [B, S, D]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.dim)

        # Final projection
        output = self.out_proj(output)

        if self.debug:
            self._debug_print("Final output", output)

        return output


class OptimizedTagEmbedding(nn.Module):
    def __init__(self, num_tags, embedding_dim, num_heads=8, dropout=0.1):
        super().__init__()
        # Single shared embedding for all tags
        self.embedding = nn.Embedding(num_tags, embedding_dim)
        self.attention = FlashAttention(embedding_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Single importance weighting for all tags
        self.tag_importance = nn.Parameter(torch.ones(num_tags) * 0.1)

        # Projection layers for unified tag context
        self.context_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self.importance_scale = nn.Parameter(torch.tensor(0.1))
        self.context_scale = nn.Parameter(torch.tensor(1.0))
        self.debug = False

    def _debug_print(self, name, tensor, extra_info=None):
        """Memory efficient debug printing with type handling"""
        if self.debug:
            print(f"\n{name}:")
            print(f"- Shape: {tensor.shape}")
            if isinstance(tensor, torch.Tensor):
                with torch.no_grad():
                    print(f"- Device: {tensor.device}")
                    print(f"- Dtype: {tensor.dtype}")

                    # Convert to float32 for statistics if needed
                    if tensor.dtype not in [torch.float16, torch.float32, torch.float64]:
                        calc_tensor = tensor.float()
                    else:
                        calc_tensor = tensor

                    try:
                        min_val = calc_tensor.min().item()
                        max_val = calc_tensor.max().item()
                        mean_val = calc_tensor.mean().item()
                        std_val = calc_tensor.std().item()
                        norm_val = torch.norm(calc_tensor).item()

                        print(f"- Value range: [{min_val:.3f}, {max_val:.3f}]")
                        print(f"- Mean: {mean_val:.3f}")
                        print(f"- Std: {std_val:.3f}")
                        print(f"- L2 Norm: {norm_val:.3f}")

                        if extra_info:
                            print(f"- Additional info: {extra_info}")
                    except Exception as e:
                        print(f"- Could not compute statistics: {str(e)}")

    def _debug_tensor(self, name, tensor):
        """Debug helper with dtype-specific analysis"""
        if self.debug and isinstance(tensor, torch.Tensor):
            print(f"\n{name}:")
            print(f"- Shape: {tensor.shape}")
            print(f"- Device: {tensor.device}")
            print(f"- Dtype: {tensor.dtype}")
            with torch.no_grad():
                has_nan = torch.isnan(tensor).any().item() if tensor.is_floating_point() else False
                has_inf = torch.isinf(tensor).any().item() if tensor.is_floating_point() else False
                print(f"- Contains NaN: {has_nan}")
                print(f"- Contains Inf: {has_inf}")

                # Different stats for different dtypes
                if tensor.is_floating_point():
                    print(f"- Range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
                    print(f"- Mean: {tensor.mean().item():.3f}")
                    print(f"- Std: {tensor.std().item():.3f}")
                else:
                    # For integer tensors
                    print(f"- Range: [{tensor.min().item()}, {tensor.max().item()}]")
                    print(f"- Unique values: {tensor.unique().numel()}")

    def _process_category(self, indices, masks):
        """Process a single category of tags"""
        # Get embeddings for this category
        embeddings = self.embedding(indices)

        if self.debug:
            self._debug_tensor("Category embeddings", embeddings)

        # Apply importance weights
        importance = torch.sigmoid(self.tag_importance) * self.importance_scale
        importance = torch.clamp(importance, min=0.01, max=10.0)
        importance_weights = importance[indices].unsqueeze(-1)

        # Apply and normalize
        embeddings = embeddings * importance_weights
        embeddings = self.norm1(embeddings)

        # Apply attention if we have more than one tag
        if embeddings.size(1) > 1:
            if masks is not None:
                attention_mask = torch.einsum('bi,bj->bij', masks, masks)
                attended = self.attention(embeddings, mask=attention_mask)
            else:
                attended = self.attention(embeddings)
            embeddings = self.norm2(attended)

        # Pool embeddings with masking
        if masks is not None:
            masked_embeddings = embeddings * masks.unsqueeze(-1)
            pooled = masked_embeddings.sum(dim=1) / masks.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            pooled = embeddings.mean(dim=1)

        return pooled, embeddings

    def forward(self, tag_indices_dict, tag_masks_dict=None):
        """
        Process all tags in a unified embedding space
        Args:
            tag_indices_dict: dict of {category: tensor of indices}
            tag_masks_dict: dict of {category: tensor of masks}
        """
        if self.debug:
            print("\nOptimizedTagEmbedding Forward Pass")

        # Concatenate all indices and masks
        all_indices = []
        all_masks = []
        batch_size = None

        for category, indices in tag_indices_dict.items():
            if batch_size is None:
                batch_size = indices.size(0)
            all_indices.append(indices)
            if tag_masks_dict:
                all_masks.append(tag_masks_dict[category])

        # Stack along sequence dimension
        combined_indices = torch.cat(all_indices, dim=1)  # [B, total_seq_len]
        if tag_masks_dict:
            combined_masks = torch.cat(all_masks, dim=1)  # [B, total_seq_len]

        if self.debug:
            self._debug_tensor("Combined indices", combined_indices)
            if tag_masks_dict:
                self._debug_tensor("Combined masks", combined_masks)

        # Get embeddings for all tags using shared embedding
        embeddings = self.embedding(combined_indices)  # [B, total_seq_len, D]

        if self.debug:
            self._debug_tensor("Base embeddings", embeddings)

        # Apply unified importance weighting
        importance = torch.sigmoid(self.tag_importance) * self.importance_scale
        importance = torch.clamp(importance, min=0.01, max=10.0)
        importance_weights = importance[combined_indices].unsqueeze(-1)

        # Apply and normalize importance weights
        embeddings = embeddings * importance_weights
        embeddings = self.norm1(embeddings)

        if self.debug:
            self._debug_tensor("Weighted embeddings", embeddings)

        # Apply attention across all tags together
        if tag_masks_dict:
            attention_mask = torch.einsum('bi,bj->bij', combined_masks, combined_masks)
            attended = self.attention(embeddings, mask=attention_mask)
        else:
            attended = self.attention(embeddings)

        attended = self.norm2(attended)

        if self.debug:
            self._debug_tensor("Attended embeddings", attended)

        # Global pooling with masking
        if tag_masks_dict:
            masked_embeddings = attended * combined_masks.unsqueeze(-1)
            tag_context = masked_embeddings.sum(dim=1) / combined_masks.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            tag_context = attended.mean(dim=1)

        # Project and scale context
        tag_context = self.context_proj(tag_context)
        context_scale = torch.clamp(self.context_scale, min=0.1, max=10.0)
        tag_context = tag_context * context_scale

        if self.debug:
            self._debug_tensor("Final tag context", tag_context)

        return tag_context, attended


class TagDataset:
    """Lightweight dataset wrapper for inference only"""

    def __init__(self, total_tags, idx_to_tag, tag_to_category):
        self.total_tags = total_tags
        self.idx_to_tag = idx_to_tag if isinstance(idx_to_tag, dict) else {int(k): v for k, v in idx_to_tag.items()}
        self.tag_to_category = tag_to_category

    def get_tag_info(self, idx):
        """Get tag name and category for a given index"""
        tag_name = self.idx_to_tag.get(idx, f"unknown-{idx}")
        category = self.tag_to_category.get(tag_name, "general")
        return tag_name, category


class ImageTagger(nn.Module):
    def __init__(self, total_tags, dataset, model_name='efficientnet_v2_l',
                 num_heads=16, dropout=0.1, pretrained=True,
                 tag_context_size=256):
        super().__init__()
        # Debug and stats flags
        self._flags = {
            'debug': False,
            'model_stats': False
        }

        # Core model config
        self.dataset = dataset
        self.tag_context_size = tag_context_size
        self.embedding_dim = 1280  # Fixed to EfficientNetV2-L output dimension

        # Initialize backbone
        if model_name == 'efficientnet_v2_l':
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

        # Tag embeddings at full dimension
        self.tag_embedding = nn.Embedding(total_tags, self.embedding_dim)
        self.tag_attention = FlashAttention(self.embedding_dim, num_heads, dropout)
        self.tag_norm = nn.LayerNorm(self.embedding_dim)

        # Improved cross attention projection
        self.cross_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        )

        # Cross attention at full dimension
        self.cross_attention = FlashAttention(self.embedding_dim, num_heads, dropout)
        self.cross_norm = nn.LayerNorm(self.embedding_dim)

        # Refined classifier with improved bottleneck
        self.refined_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2),  # Doubled input size for residual
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

    def _get_selected_tags(self, logits):
        """Select top-K tags based on prediction confidence"""
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Get top-K predictions for each image in batch
        batch_size = logits.size(0)
        topk_values, topk_indices = torch.topk(
            probs, k=self.tag_context_size, dim=1, largest=True, sorted=True
        )

        return topk_indices, topk_values

    @property
    def debug(self):
        return self._flags['debug']

    @debug.setter
    def debug(self, value):
        self._flags['debug'] = value

    @property
    def model_stats(self):
        return self._flags['model_stats']

    @model_stats.setter
    def model_stats(self, value):
        self._flags['model_stats'] = value

    def preprocess_image(self, image_path, image_size=512):
        """Process an image for inference using same preprocessing as training"""
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found at path: {image_path}")

        # Initialize the same transform used during training
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        try:
            with Image.open(image_path) as img:
                # Convert RGBA or Palette images to RGB
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                # Get original dimensions
                width, height = img.size
                aspect_ratio = width / height

                # Calculate new dimensions to maintain aspect ratio
                if aspect_ratio > 1:
                    new_width = image_size
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = image_size
                    new_width = int(new_height * aspect_ratio)

                # Resize with LANCZOS filter
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Create new image with padding
                new_image = Image.new('RGB', (image_size, image_size), (0, 0, 0))
                paste_x = (image_size - new_width) // 2
                paste_y = (image_size - new_height) // 2
                new_image.paste(img, (paste_x, paste_y))

                # Apply transforms (without normalization)
                img_tensor = transform(new_image)
                return img_tensor
        except Exception as e:
            raise Exception(f"Error processing {image_path}: {str(e)}")

    def forward(self, x):
        """Forward pass with simplified feature handling"""
        # Initialize tracking dicts
        model_stats = {} if self.model_stats else {}
        debug_tensors = {} if self.debug else None

        # 1. Image Feature Extraction
        features = self.backbone.features(x)
        features = self.spatial_pool(features).squeeze(-1).squeeze(-1)

        # 2. Initial Tag Predictions
        initial_logits = self.initial_classifier(features)
        initial_preds = torch.clamp(initial_logits / self.temperature, min=-15.0, max=15.0)

        # 3. Tag Selection & Embedding (simplified)
        pred_tag_indices, _ = self._get_selected_tags(initial_preds)
        tag_embeddings = self.tag_embedding(pred_tag_indices)

        # 4. Self-Attention on Tags
        attended_tags = self.tag_attention(tag_embeddings)
        attended_tags = self.tag_norm(attended_tags)

        # 5. Cross-Attention between Features and Tags
        features_proj = self.cross_proj(features)
        features_expanded = features_proj.unsqueeze(1).expand(-1, self.tag_context_size, -1)

        cross_attended = self.cross_attention(features_expanded, attended_tags)
        cross_attended = self.cross_norm(cross_attended)

        # 6. Feature Fusion with Residual Connection
        fused_features = cross_attended.mean(dim=1)  # Average across tag dimension
        # Concatenate original and attended features
        combined_features = torch.cat([features, fused_features], dim=-1)

        # 7. Refined Predictions
        refined_logits = self.refined_classifier(combined_features)
        refined_preds = torch.clamp(refined_logits / self.temperature, min=-15.0, max=15.0)

        # Return both prediction sets
        return initial_preds, refined_preds

    def predict(self, image_path, threshold=0.325, category_thresholds=None):
        """
        Run inference on an image with support for category-specific thresholds.
        """
        # Preprocess the image
        img_tensor = self.preprocess_image(image_path).unsqueeze(0)

        # Move to the same device as model and convert to half precision
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype  # Match model's precision
        img_tensor = img_tensor.to(device, dtype=dtype)

        # Run inference
        with torch.no_grad():
            initial_preds, refined_preds = self.forward(img_tensor)

            # Apply sigmoid to get probabilities
            initial_probs = torch.sigmoid(initial_preds)
            refined_probs = torch.sigmoid(refined_preds)

            # Apply thresholds
            if category_thresholds:
                # Create binary prediction tensors
                refined_binary = torch.zeros_like(refined_probs)

                # Apply thresholds by category
                for category, cat_threshold in category_thresholds.items():
                    # Create a mask for tags in this category
                    category_mask = torch.zeros_like(refined_probs, dtype=torch.bool)

                    # Find indices for this category
                    for tag_idx in range(refined_probs.size(-1)):
                        try:
                            _, tag_category = self.dataset.get_tag_info(tag_idx)
                            if tag_category == category:
                                category_mask[:, tag_idx] = True
                        except:
                            continue

                    # Apply threshold only to tags in this category - ensure dtype consistency
                    cat_threshold_tensor = torch.tensor(cat_threshold, device=device, dtype=dtype)
                    refined_binary[category_mask] = (refined_probs[category_mask] >= cat_threshold_tensor).to(dtype)

                predictions = refined_binary
            else:
                # Use the same threshold for all tags
                threshold_tensor = torch.tensor(threshold, device=device, dtype=dtype)
                predictions = (refined_probs >= threshold_tensor).to(dtype)

            # Return both probabilities and thresholded predictions
            return {
                'initial_probabilities': initial_probs,
                'refined_probabilities': refined_probs,
                'predictions': predictions
            }

    def get_tags_from_predictions(self, predictions, include_probabilities=True):
        """
        Convert model predictions to human-readable tags grouped by category.
        """
        # Get non-zero predictions
        if predictions.dim() > 1:
            predictions = predictions[0]  # Remove batch dimension

        # Get indices of positive predictions
        indices = torch.where(predictions > 0)[0].cpu().tolist()

        # Group by category
        result = {}
        for idx in indices:
            tag_name, category = self.dataset.get_tag_info(idx)

            if category not in result:
                result[category] = []

            if include_probabilities:
                prob = predictions[idx].item()
                result[category].append((tag_name, prob))
            else:
                result[category].append(tag_name)

        # Sort tags by probability within each category
        if include_probabilities:
            for category in result:
                result[category] = sorted(result[category], key=lambda x: x[1], reverse=True)

        return result


def load_model(model_dir, device='cuda'):
    """Load model with better error handling and warnings"""
    print(f"Loading model from {model_dir}")

    try:
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load model info
        model_info_path = os.path.join(model_dir, "model_info_initial_only.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
        else:
            print("WARNING: Model info file not found, using default settings")
            model_info = {
                "tag_context_size": 256,
                "num_heads": 16,
                "precision": "float16"
            }

        # Create dataset wrapper
        dataset = TagDataset(
            total_tags=metadata['total_tags'],
            idx_to_tag=metadata['idx_to_tag'],
            tag_to_category=metadata['tag_to_category']
        )

        # Initialize model with exact settings from model_info
        model = ImageTagger(
            total_tags=metadata['total_tags'],
            dataset=dataset,
            num_heads=model_info.get('num_heads', 16),
            tag_context_size=model_info.get('tag_context_size', 256),
            pretrained=False
        )

        # Load weights
        state_dict_path = os.path.join(model_dir, "model.pt")
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"Model state dict not found at {state_dict_path}")

        state_dict = torch.load(state_dict_path, map_location=device)

        # First try strict loading
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✓ Model state dict loaded with strict=True successfully")
        except Exception as e:
            print(f"! Strict loading failed: {str(e)}")
            print("Attempting non-strict loading...")

            # Try non-strict loading
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            print(f"Non-strict loading completed with:")
            print(f"- {len(missing_keys)} missing keys")
            print(f"- {len(unexpected_keys)} unexpected keys")

            if len(missing_keys) > 0:
                print(f"Sample missing keys: {missing_keys[:5]}")
            if len(unexpected_keys) > 0:
                print(f"Sample unexpected keys: {unexpected_keys[:5]}")

        # Move model to device
        model = model.to(device)

        # Set to half precision if needed
        if model_info.get('precision') == 'float16':
            model = model.half()
            print("✓ Model converted to half precision")

        # Set to eval mode
        model.eval()
        print("✓ Model set to evaluation mode")

        # Verify parameter dtype
        param_dtype = next(model.parameters()).dtype
        print(f"✓ Model loaded with precision: {param_dtype}")

        return model, dataset

    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# Example usage
# if __name__ == "__main__":
#     import sys
#
#     # Get model directory from command line or use default
#     model_dir = sys.argv[1] if len(sys.argv) > 1 else "./exported_model"
#
#     # Load model
#     model, dataset, thresholds = load_model(model_dir)
#
#     # Display info
#     print(f"\nModel information:")
#     print(f"  Total tags: {dataset.total_tags}")
#     print(f"  Device: {next(model.parameters()).device}")
#     print(f"  Precision: {next(model.parameters()).dtype}")
#
#     # Test on an image if provided
#     if len(sys.argv) > 2:
#         image_path = sys.argv[2]
#         print(f"\nRunning inference on {image_path}")
#
#         # Use category thresholds if available
#         if thresholds and 'categories' in thresholds:
#             category_thresholds = {cat: opt['balanced']['threshold']
#                                    for cat, opt in thresholds['categories'].items()}
#             results = model.predict(image_path, category_thresholds=category_thresholds)
#         else:
#             results = model.predict(image_path)
#
#         # Get tags
#         tags = model.get_tags_from_predictions(results['predictions'])
#
#         # Print tags by category
#         print("\nPredicted tags:")
#         for category, category_tags in tags.items():
#             print(f"\n{category.capitalize()}:")
#             for tag, prob in category_tags:
#                 print(f"  {tag}: {prob:.3f}")

if __name__ == '__main__':
    safetensors_path = hf_hub_download(
        repo_id='Camais03/camie-tagger',
        repo_type='model',
        filename='model_initial.safetensors'
    )
    state_dict = load_file(safetensors_path, device='cpu')
    # state_dict = torch.load(weights_path, map_location="cpu")
    # Instantiate the model with the same parameters as training
    model = InitialOnlyImageTagger(total_tags=70527, dataset=None, pretrained=True)  # dataset not needed for forward
    model.load_state_dict(state_dict)
    model.eval()  # set to evaluation mode

    print(model)

    # # Define example input – a dummy image tensor of the expected input shape (1, 3, 512, 512)
    # dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    #
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
