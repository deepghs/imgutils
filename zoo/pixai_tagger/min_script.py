import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download

from imgutils.preprocess import parse_torchvision_transforms


class TaggingHead(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.head = torch.nn.Sequential(torch.nn.Linear(input_dim, num_classes))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        logits = self.head(x)
        probs = self.sigmoid(logits)
        return probs


def get_tags(tags_file: Path) -> tuple[dict[str, int], int, int]:
    with tags_file.open("r", encoding="utf-8") as f:
        tag_info = json.load(f)
    tag_map = tag_info["tag_map"]
    tag_split = tag_info["tag_split"]
    gen_tag_count = tag_split["gen_tag_count"]
    character_tag_count = tag_split["character_tag_count"]
    return tag_map, gen_tag_count, character_tag_count


def get_character_ip_mapping(mapping_file: Path):
    with mapping_file.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    return mapping


def get_encoder():
    base_model_repo = "hf_hub:SmilingWolf/wd-eva02-large-tagger-v3"
    encoder = timm.create_model(base_model_repo, pretrained=False)
    encoder.reset_classifier(0)
    return encoder


def get_decoder():
    decoder = TaggingHead(1024, 13461)
    return decoder


def get_model():
    encoder = get_encoder()
    decoder = get_decoder()
    model = torch.nn.Sequential(encoder, decoder)
    return model


def load_model(weights_file, device):
    model = get_model()
    states_dict = torch.load(weights_file, map_location=device, weights_only=True)
    model.load_state_dict(states_dict)
    model.to(device)
    model.eval()
    return model


def pure_pil_alpha_to_color_v2(
        image: Image.Image, color: tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Convert a PIL image with an alpha channel to a RGB image.
    This is a workaround for the fact that the model expects a RGB image, but the image may have an alpha channel.
    This function will convert the image to a RGB image, and fill the alpha channel with the given color.
    The alpha channel is the 4th channel of the image.
    """
    image.load()  # needed for split()
    background = Image.new("RGB", image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


def pil_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        image = pure_pil_alpha_to_color_v2(image)
    elif image.mode == "P":
        image = pure_pil_alpha_to_color_v2(image.convert("RGBA"))
    else:
        image = image.convert("RGB")
    return image


class EndpointHandler:
    def __init__(self, repo_id: str = 'pixai-labs/pixai-tagger-v0.9'):
        weights_file = Path(hf_hub_download(
            repo_id=repo_id,
            repo_type='model',
            filename="model_v0.9.pth",
        ))
        tags_file = Path(hf_hub_download(
            repo_id=repo_id,
            repo_type='model',
            filename="tags_v0.9_13k.json",
        ))
        mapping_file = Path(hf_hub_download(
            repo_id=repo_id,
            repo_type='model',
            filename="char_ip_map.json",
        ))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(str(weights_file), self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.fetch_image_timeout = 5.0
        self.default_general_threshold = 0.3
        self.default_character_threshold = 0.85

        tag_map, self.gen_tag_count, self.character_tag_count = get_tags(tags_file)

        # Invert the tag_map for efficient index-to-tag lookups
        self.index_to_tag_map = {v: k for k, v in tag_map.items()}

        self.character_ip_mapping = get_character_ip_mapping(mapping_file)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        inputs = data.pop("inputs", data)

        fetch_start_time = time.time()
        if isinstance(inputs, Image.Image):
            image = inputs
        elif image_url := inputs.pop("url", None):
            with requests.get(
                    image_url, stream=True, timeout=self.fetch_image_timeout
            ) as res:
                res.raise_for_status()
                image = Image.open(res.raw)
        elif image_base64_encoded := inputs.pop("image", None):
            image = Image.open(io.BytesIO(base64.b64decode(image_base64_encoded)))
        else:
            raise ValueError(f"No image or url provided: {data}")
        # remove alpha channel if it exists
        image = pil_to_rgb(image)
        fetch_time = time.time() - fetch_start_time

        parameters = data.pop("parameters", {})
        general_threshold = parameters.pop(
            "general_threshold", self.default_general_threshold
        )
        character_threshold = parameters.pop(
            "character_threshold", self.default_character_threshold
        )

        inference_start_time = time.time()
        with torch.inference_mode():
            # Preprocess image on CPU, then pin memory for faster async transfer
            image_tensor = self.transform(image).unsqueeze(0).pin_memory()

            # Asynchronously move image to GPU
            image_tensor = image_tensor.to(self.device, non_blocking=True)

            # Run model on GPU
            probs = self.model(image_tensor)[0]  # Get probs for the single image

            # Perform thresholding directly on the GPU
            general_mask = probs[: self.gen_tag_count] > general_threshold
            character_mask = probs[self.gen_tag_count:] > character_threshold

            # Get the indices of positive tags on the GPU
            general_indices = general_mask.nonzero(as_tuple=True)[0]
            character_indices = (
                    character_mask.nonzero(as_tuple=True)[0] + self.gen_tag_count
            )

            # Combine indices and move the small result tensor to the CPU
            combined_indices = torch.cat((general_indices, character_indices)).cpu()

        inference_time = time.time() - inference_start_time

        post_process_start_time = time.time()

        cur_gen_tags = []
        cur_char_tags = []

        # Use the efficient pre-computed map for lookups
        for i in combined_indices:
            idx = i.item()
            tag = self.index_to_tag_map[idx]
            if idx < self.gen_tag_count:
                cur_gen_tags.append(tag)
            else:
                cur_char_tags.append(tag)

        ip_tags = []
        for tag in cur_char_tags:
            if tag in self.character_ip_mapping:
                ip_tags.extend(self.character_ip_mapping[tag])
        ip_tags = sorted(set(ip_tags))
        post_process_time = time.time() - post_process_start_time

        logging.info(
            f"Timing - Fetch: {fetch_time:.3f}s, Inference: {inference_time:.3f}s, Post-process: {post_process_time:.3f}s, Total: {fetch_time + inference_time + post_process_time:.3f}s"
        )

        return {
            "feature": cur_gen_tags,
            "character": cur_char_tags,
            "ip": ip_tags,
        }


if __name__ == '__main__':
    handler = EndpointHandler()
    print(handler.transform)
    print(parse_torchvision_transforms(handler.transform))
