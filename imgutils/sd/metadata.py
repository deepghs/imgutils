"""
Overview:
    Metadata parser and formatter for a1111's webui generated images.

    Here are 2 sample images: :download:`sd_metadata_simple.png <sd_metadata_simple.png>`
    and :download:`sd_metadata_complex.png <sd_metadata_complex.png>`.
"""
import io
import json
import mimetypes
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ..data import ImageTyping, load_image
from ..metadata import read_geninfo_parameters, read_geninfo_exif, write_geninfo_exif, write_geninfo_gif, \
    read_geninfo_gif

_PARAM_PATTERN = re.compile(r'\s*(?P<key>[\w ]+):\s*(?P<value>"(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
_SIZE_PATTERN = re.compile(r"^(?P<size1>-?\d+)\s*x\s*(?P<size2>-?\d+)$")
_NEG_LINE_PATTERN = re.compile(r'^Negative prompt:\s*(?P<content>[\S\s]*?)$')

_PRE_KEYS = [
    "Steps", "Sampler", "CFG scale", "Image CFG scale", "Seed", "Face restoration", "Size",
    "Model hash", "Model", "VAE hash", "VAE", "Variation seed", "Variation seed strength",
    "Seed resize from", "Denoising strength", "Conditional mask weight", "Clip skip", "ENSD",
    "Token merging ratio", "Token merging ratio hr", "Init image hash", "RNG", "NGMS", "Tiling",
]
_POST_KEYS = [
    "Version", "User"
]


def _sdmeta_quote(value):
    if isinstance(value, tuple) and len(value) == 2:
        return f'{value[0]}x{value[1]}'
    if ',' not in str(value) and '\n' not in str(value) and ':' not in str(value):
        return value

    return json.dumps(value, ensure_ascii=False)


@dataclass
class SDMetaData:
    """
    Store information parsed from the metadata of a PNG image.

    :param prompt: The main prompt text.
    :type prompt: str
    :param neg_prompt: The negative prompt text.
    :type neg_prompt: str
    :param parameters: A dictionary containing various parameters.
    :type parameters: Dict[str, Any]
    """

    prompt: str
    neg_prompt: str
    parameters: Dict[str, Any]

    def __str__(self):
        """
        Get a string representation of the metadata.

        :return: A string representation of the metadata.
        :rtype: str

        Examples::
            >>> from imgutils.sd import get_sdmeta_from_image
            >>>
            >>> sd1 = get_sdmeta_from_image('sd_metadata_simple.png')
            >>> print(sd1)
            (extremely delicate and beautiful), best quality, official art, global illumination, soft shadow, super detailed, Japanese light novel cover, 4K, metal_texture, (striped_background), super detailed background, more detailed, rich detailed, extremely detailed CG unity 8k wallpaper, ((unreal)), sci-fi,(fantasy),(masterpiece),(super delicate), (illustration), (extremely delicate and beautiful), anime coloring,
            (silver_skin), ((high-cut silver_impossible_bodysuit), ((gem_on_chest)),(high-cut_silver_mechanical_leotard)),headgear,
            (focus-on:1.1),(1_girl),((solo)),slim_waist,white hair, long hair, luminous yellow eyes,(medium_breast:1.2), (Indistinct_cameltoe:0.9), (flat_crotch:1.1),(coquettish), (squint:1.4),(evil_smile :1.35),(dark_persona), [open mouth: 0.7], standing,[wet:0.7],
            slim_face, tall_girl,(mature),mature_face, (slim_figure), (slim_legs:1.1), (groin:1.1), ((bare_thighs)),
            Negative prompt: EasyNegative, sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), ((2girl)), (deformed fingers:1.2), (long fingers:1.2),(bad-artist-anime), bad-artist, bad hand, blush, (lipstick),skindentation, tie, ((big_breast)), (nipple), thighhighs, pubic_hair, pussy, black and white,(3d), ((realistic)),blurry,nipple slip, (nipple), blush, head_out_of_frame,curvy,
            Steps: 20, Sampler: DDIM, CFG scale: 7, Seed: 3827064803, Size: 512x848, Model hash: eb49192009, Model: AniDosMix, Clip skip: 2
        """
        return self._sdmeta_text()

    def _sdmeta_text(self):
        """
        Generate a formatted string representation of the metadata.

        This internal method is used by __str__ and other methods to create a consistent
        string representation of the metadata.

        :return: A formatted string containing the metadata.
        :rtype: str
        """
        with io.StringIO() as sio:
            print(self.prompt, file=sio)
            if self.neg_prompt:
                print(f'Negative prompt: {self.neg_prompt}', file=sio)

            _ext_keys = set(self.parameters.keys()) - set(_PRE_KEYS) - set(_POST_KEYS)
            _params = {}
            for key in _PRE_KEYS:
                if key in self.parameters:
                    _params[key] = self.parameters[key]
            for key in self.parameters.keys():
                if key in _ext_keys:
                    _params[key] = self.parameters[key]
            for key in _POST_KEYS:
                if key in self.parameters:
                    _params[key] = self.parameters[key]

            if _params:
                print(", ".join([
                    k if k == v else f'{k}: {_sdmeta_quote(v)}'
                    for k, v in _params.items() if v is not None
                ]), file=sio)

            return sio.getvalue().strip()

    @property
    def text(self) -> str:
        """
        Get the metadata as a formatted string.

        This property provides a convenient way to access the string representation
        of the metadata without calling _sdmeta_text() directly.

        :return: A formatted string containing the metadata.
        :rtype: str

        Example:
            >>> metadata = SDMetaData(
            ...     prompt="A starry night",
            ...     neg_prompt="Daylight",
            ...     parameters={"Steps": 40, "Sampler": "Euler", "CFG scale": 8}
            ... )
            >>> print(metadata.text)
            A starry night
            Negative prompt: Daylight
            Steps: 40, Sampler: Euler, CFG scale: 8
        """
        return self._sdmeta_text()

    @property
    def pnginfo(self) -> PngInfo:
        """
        Generate a PngInfo object with the metadata.

        This can be used when saving an image with custom metadata.

        :return: A PngInfo object containing the metadata.
        :rtype: PngInfo

        Examples::
            >>> from PIL import Image
            >>> from imgutils.sd import get_sdmeta_from_image
            >>>
            >>> # get metadata
            >>> sd1 = get_sdmeta_from_image('sd_metadata_simple.png')
            >>>
            >>> # create an image
            >>> img = Image.new('RGB', (256, 256), 'white')
            >>>
            >>> # save this image with sd1's metadata
            >>> img.save('new_image.png', pnginfo=sd1.pnginfo)
            >>>
            >>> # let's see what is in 'new_image.png'
            >>> get_sdmeta_from_image('new_image.png')
            SDMetaData(prompt='(extremely delicate and beautiful), best quality, official art, global illumination, soft shadow, super detailed, Japanese light novel cover, 4K, metal_texture, (striped_background), super detailed background, more detailed, rich detailed, extremely detailed CG unity 8k wallpaper, ((unreal)), sci-fi,(fantasy),(masterpiece),(super delicate), (illustration), (extremely delicate and beautiful), anime coloring,\\n(silver_skin), ((high-cut silver_impossible_bodysuit), ((gem_on_chest)),(high-cut_silver_mechanical_leotard)),headgear,\\n(focus-on:1.1),(1_girl),((solo)),slim_waist,white hair, long hair, luminous yellow eyes,(medium_breast:1.2), (Indistinct_cameltoe:0.9), (flat_crotch:1.1),(coquettish), (squint:1.4),(evil_smile :1.35),(dark_persona), [open mouth: 0.7], standing,[wet:0.7],\\nslim_face, tall_girl,(mature),mature_face, (slim_figure), (slim_legs:1.1), (groin:1.1), ((bare_thighs)),', neg_prompt='EasyNegative, sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), ((2girl)), (deformed fingers:1.2), (long fingers:1.2),(bad-artist-anime), bad-artist, bad hand, blush, (lipstick),skindentation, tie, ((big_breast)), (nipple), thighhighs, pubic_hair, pussy, black and white,(3d), ((realistic)),blurry,nipple slip, (nipple), blush, head_out_of_frame,curvy,', parameters={'Steps': 20, 'Sampler': 'DDIM', 'CFG scale': 7, 'Seed': 3827064803, 'Size': (512, 848), 'Model hash': 'eb49192009', 'Model': 'AniDosMix', 'Clip skip': 2})
        """
        info = PngInfo()
        info.add_text('parameters', self._sdmeta_text())
        return info


def _parse_parameters(param_text: str):
    params = {}
    for key, value in _PARAM_PATTERN.findall(param_text):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass

        if isinstance(value, str):
            size_matching = _SIZE_PATTERN.match(value)
            if size_matching is not None:
                params[key] = (
                    int(size_matching.group('size1')),
                    int(size_matching.group('size2'))
                )
            else:
                params[key] = value
        else:
            params[key] = value

    return params


def parse_sdmeta_from_text(x: str) -> SDMetaData:
    """
    Parse metadata information from a string.

    :param x: The input string containing metadata.
    :type x: str
    :return: A SDMetaData object containing the parsed metadata.
    :rtype: SDMetaData

    Examples::
        >>> from imgutils.sd import parse_sdmeta_from_text
        >>>
        >>> sd1 = parse_sdmeta_from_text(\"\"\"
        ... (extremely delicate and beautiful), best quality, official art, global illumination, soft shadow, super detailed, Japanese light novel cover, 4K, metal_texture, (striped_background), super detailed background, more detailed, rich detailed, extremely detailed CG unity 8k wallpaper, ((unreal)), sci-fi,(fantasy),(masterpiece),(super delicate), (illustration), (extremely delicate and beautiful), anime coloring,
        ... (silver_skin), ((high-cut silver_impossible_bodysuit), ((gem_on_chest)),(high-cut_silver_mechanical_leotard)),headgear,
        ... (focus-on:1.1),(1_girl),((solo)),slim_waist,white hair, long hair, luminous yellow eyes,(medium_breast:1.2), (Indistinct_cameltoe:0.9), (flat_crotch:1.1),(coquettish), (squint:1.4),(evil_smile :1.35),(dark_persona), [open mouth: 0.7], standing,[wet:0.7],
        ... slim_face, tall_girl,(mature),mature_face, (slim_figure), (slim_legs:1.1), (groin:1.1), ((bare_thighs)),
        ... Negative prompt: EasyNegative, sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), ((2girl)), (deformed fingers:1.2), (long fingers:1.2),(bad-artist-anime), bad-artist, bad hand, blush, (lipstick),skindentation, tie, ((big_breast)), (nipple), thighhighs, pubic_hair, pussy, black and white,(3d), ((realistic)),blurry,nipple slip, (nipple), blush, head_out_of_frame,curvy,
        ... Steps: 20, Sampler: DDIM, CFG scale: 7, Seed: 3827064803, Size: 512x848, Model hash: eb49192009, Model: AniDosMix, Clip skip: 2
        ... \"\"\")
        >>> sd1
        SDMetaData(prompt='(extremely delicate and beautiful), best quality, official art, global illumination, soft shadow, super detailed, Japanese light novel cover, 4K, metal_texture, (striped_background), super detailed background, more detailed, rich detailed, extremely detailed CG unity 8k wallpaper, ((unreal)), sci-fi,(fantasy),(masterpiece),(super delicate), (illustration), (extremely delicate and beautiful), anime coloring,\\n(silver_skin), ((high-cut silver_impossible_bodysuit), ((gem_on_chest)),(high-cut_silver_mechanical_leotard)),headgear,\\n(focus-on:1.1),(1_girl),((solo)),slim_waist,white hair, long hair, luminous yellow eyes,(medium_breast:1.2), (Indistinct_cameltoe:0.9), (flat_crotch:1.1),(coquettish), (squint:1.4),(evil_smile :1.35),(dark_persona), [open mouth: 0.7], standing,[wet:0.7],\\nslim_face, tall_girl,(mature),mature_face, (slim_figure), (slim_legs:1.1), (groin:1.1), ((bare_thighs)),', neg_prompt='EasyNegative, sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), ((2girl)), (deformed fingers:1.2), (long fingers:1.2),(bad-artist-anime), bad-artist, bad hand, blush, (lipstick),skindentation, tie, ((big_breast)), (nipple), thighhighs, pubic_hair, pussy, black and white,(3d), ((realistic)),blurry,nipple slip, (nipple), blush, head_out_of_frame,curvy,', parameters={'Steps': 20, 'Sampler': 'DDIM', 'CFG scale': 7, 'Seed': 3827064803, 'Size': (512, 848), 'Model hash': 'eb49192009', 'Model': 'AniDosMix', 'Clip skip': 2})
        >>> type(sd1)
        <class 'imgutils.sd.metadata.SDMetaData'>
        >>>
        >>> sd2 = parse_sdmeta_from_text(\"\"\"
        ... 1girl, solo, blue eyes, black footwear, white hair, looking at viewer, shoes, full body, standing, bangs, indoors, wide sleeves, ahoge, dress, closed mouth, blush, long sleeves, potted plant, bag, plant, hair bun, window,<lora:BlueArchive10:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1>,BlueArchive,
        ... Negative prompt: Neg1,Negative,
        ... Steps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 7, Seed: 2647703743, Size: 768x768, Model hash: 72bd94132e, Model: CuteMix, Denoising strength: 0.7, ControlNet 0: "preprocessor: openpose, model: control_v11p_sd15_openpose [cab727d4], weight: 1, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: False, control mode: Balanced, preprocessor params: (512, 64, 64)", Hires upscale: 2, Hires upscaler: Latent, TI hashes: "Neg1: 339cc9210f70, Negative: 66a7279a88dd", Version: v1.5.1
        ... \"\"\")
        >>> sd2
        SDMetaData(prompt='1girl, solo, blue eyes, black footwear, white hair, looking at viewer, shoes, full body, standing, bangs, indoors, wide sleeves, ahoge, dress, closed mouth, blush, long sleeves, potted plant, bag, plant, hair bun, window,<lora:BlueArchive10:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1>,BlueArchive,', neg_prompt='Neg1,Negative,', parameters={'Steps': 20, 'Sampler': 'DPM++ 2M SDE Karras', 'CFG scale': 7, 'Seed': 2647703743, 'Size': (768, 768), 'Model hash': '72bd94132e', 'Model': 'CuteMix', 'Denoising strength': 0.7, 'ControlNet 0': 'preprocessor: openpose, model: control_v11p_sd15_openpose [cab727d4], weight: 1, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: False, control mode: Balanced, preprocessor params: (512, 64, 64)', 'Hires upscale': 2, 'Hires upscaler': 'Latent', 'TI hashes': 'Neg1: 339cc9210f70, Negative: 66a7279a88dd', 'Version': 'v1.5.1'})
        >>> type(sd2)
        <class 'imgutils.sd.metadata.SDMetaData'>
    """
    x = textwrap.dedent(x).strip()
    all_lines = x.splitlines(keepends=False)
    if len(all_lines) >= 1:
        prompt_lines = all_lines[:-1]
        argument_line = all_lines[-1]
    else:
        prompt_lines = []
        argument_line = ''
    if len(_PARAM_PATTERN.findall(argument_line)) < 3:
        prompt_lines.append(argument_line)
        argument_line = ''

    # 0x1 means prompt, 0x2 means neg prompt
    status = 0x1
    with io.StringIO() as i_prompt, io.StringIO() as i_neg_prompt:
        for line in prompt_lines:
            line = line.strip()
            neg_matching = _NEG_LINE_PATTERN.fullmatch(line)
            if neg_matching:
                line = neg_matching.group('content').strip()
                status = 0x2

            if status == 0x1:
                print(line, file=i_prompt)
            elif status == 0x2:
                print(line, file=i_neg_prompt)
            else:
                # if this line is triggered, this state machine must be buggy
                raise ValueError(f'Unknown parsing status - {status!r}.')  # pragma: no cover

        prompt = i_prompt.getvalue().strip()
        neg_prompt = i_neg_prompt.getvalue().strip()

    params = _parse_parameters(argument_line)
    return SDMetaData(prompt, neg_prompt, params)


class _InvalidSDMetaError(Exception):
    """
    Custom exception raised when SD metadata is invalid or not found.

    This exception is used internally to signal that the metadata
    validation or extraction process has failed.
    """
    pass


def _sdtext_validate(text: str) -> str:
    """
    Validate the given text as SD metadata.

    This function attempts to validate the input text as SD metadata. It first tries
    to validate it as NAI (Novel AI) metadata, and if that fails, it checks if the
    text is non-empty. If both checks fail, it raises an _InvalidSDMetaError.

    :param text: The text to validate as SD metadata.
    :type text: str

    :return: The validated text if it passes the checks.
    :rtype: str

    :raises _InvalidSDMetaError: If the text is invalid or empty.
    """
    from .nai import _naimeta_text_validate, _InvalidNAIMetaError

    try:
        _naimeta_text_validate(text)
    except _InvalidNAIMetaError:
        pass
    else:
        raise _InvalidSDMetaError

    if text:
        return text
    else:
        raise _InvalidSDMetaError


def _get_raw_sdtext(image: ImageTyping) -> Optional[str]:
    """
    Extract raw SD metadata text from the given image.

    This function attempts to read SD metadata from various sources within the image,
    including PNG info, EXIF data, and GIF metadata. It tries each method in turn
    and returns the first valid SD metadata text found.

    :param image: The input image to extract metadata from.
    :type image: ImageTyping

    :return: The raw SD metadata text if found, otherwise None.
    :rtype: Optional[str]

    :raises _InvalidSDMetaError: If no valid SD metadata is found in the image.
    """
    image = load_image(image, force_background=None, mode=None)

    try:
        return _sdtext_validate(read_geninfo_parameters(image))
    except _InvalidSDMetaError:
        pass

    try:
        return _sdtext_validate(read_geninfo_exif(image))
    except _InvalidSDMetaError:
        pass

    try:
        return _sdtext_validate(read_geninfo_gif(image))
    except _InvalidSDMetaError:
        raise _InvalidSDMetaError


def get_sdmeta_from_image(image: ImageTyping) -> Optional[SDMetaData]:
    """
    Extract and parse Stable Diffusion metadata from an image.

    This function attempts to read SD metadata from various sources within the image,
    including PNG info, EXIF data, and GIF metadata. If found, it parses the metadata
    into an SDMetaData object.

    :param image: The input image, which can be a file path, URL, or PIL Image object.
    :type image: ImageTyping

    :return: An SDMetaData object containing the parsed metadata if available, else None.
    :rtype: Optional[SDMetaData]

    :raises: Various exceptions may be raised by the underlying image loading and
             metadata reading functions.

    Example usage:
        >>> from imgutils.sd import get_sdmeta_from_image
        >>> sd_meta = get_sdmeta_from_image('path/to/image.png')
        >>> if sd_meta:
        ...     print(f"Prompt: {sd_meta.prompt}")
        ...     print(f"Negative prompt: {sd_meta.neg_prompt}")
        ...     print(f"Parameters: {sd_meta.parameters}")
        ... else:
        ...     print("No SD metadata found in the image.")

    Note: This function depends on the load_image and parse_sdmeta_from_text functions.
    Ensure these are properly imported or defined in the current scope.
    """
    image = load_image(image, mode=None, force_background=None)
    try:
        pnginfo_text = _get_raw_sdtext(image)
    except _InvalidSDMetaError:
        return None
    else:
        return parse_sdmeta_from_text(pnginfo_text)


def _save_png_with_sdmeta(image: Image.Image, dst_file: Union[str, os.PathLike], metadata: SDMetaData, **kwargs):
    """
    Internal function to save a PNG image with SD metadata.

    :param image: The PIL Image object to save.
    :param dst_file: The destination file path.
    :param metadata: The SDMetaData object containing the metadata to save.
    :param kwargs: Additional keyword arguments to pass to the PIL save function.
    """
    image.save(dst_file, pnginfo=metadata.pnginfo, **kwargs)


def _save_exif_with_sdmeta(image: Image.Image, dst_file: Union[str, os.PathLike], metadata: SDMetaData, **kwargs):
    """
    Internal function to save an image with SD metadata in EXIF format.

    :param image: The PIL Image object to save.
    :param dst_file: The destination file path.
    :param metadata: The SDMetaData object containing the metadata to save.
    :param kwargs: Additional keyword arguments to pass to the write_geninfo_exif function.
    """
    write_geninfo_exif(image, dst_file, metadata.text, **kwargs)


def _save_gif_with_sdmeta(image: Image.Image, dst_file: Union[str, os.PathLike], metadata: SDMetaData, **kwargs):
    """
    Internal function to save a GIF image with SD metadata.

    :param image: The PIL Image object to save.
    :param dst_file: The destination file path.
    :param metadata: The SDMetaData object containing the metadata to save.
    :param kwargs: Additional keyword arguments to pass to the write_geninfo_gif function.
    """
    write_geninfo_gif(image, dst_file, metadata.text, **kwargs)


_FN_IMG_SAVE = {
    'image/png': _save_png_with_sdmeta,
    'image/jpeg': _save_exif_with_sdmeta,
    'image/webp': _save_exif_with_sdmeta,
    'image/gif': _save_gif_with_sdmeta,
}


def save_image_with_sdmeta(image: ImageTyping, dst_file: Union[str, os.PathLike], metadata: SDMetaData, **kwargs):
    """
    Save an image with Stable Diffusion metadata.

    This function saves the given image to the specified destination file, including
    the provided SD metadata. The metadata is saved in a format appropriate for the
    output image type (PNG, JPEG, WebP, or GIF).

    :param image: The input image, which can be a file path, URL, or PIL Image object.
    :type image: ImageTyping
    :param dst_file: The destination file path where the image will be saved.
    :type dst_file: Union[str, os.PathLike]
    :param metadata: The SD metadata to include with the image.
    :type metadata: SDMetaData
    :param kwargs: Additional keyword arguments to pass to the underlying save function.

    :raises SystemError: If the output file type is not supported for saving with metadata.
    :raises: Various exceptions may be raised by the underlying image loading and
             saving functions.

    Example usage:
        >>> from imgutils.sd import get_sdmeta_from_image, save_image_with_sdmeta
        >>> input_image = 'path/to/input.png'
        >>> output_image = 'path/to/output.png'
        >>> sd_meta = get_sdmeta_from_image(input_image)
        >>> if sd_meta:
        ...     save_image_with_sdmeta(input_image, output_image, sd_meta)
        ...     print(f"Image saved with SD metadata to {output_image}")
        ... else:
        ...     print("No SD metadata found in the input image.")
    """
    mimetype, _ = mimetypes.guess_type(str(dst_file))
    if mimetype not in _FN_IMG_SAVE:
        raise SystemError(f'Not supported to save as a {mimetype!r} type, '
                          f'supported mimetypes are {sorted(_FN_IMG_SAVE.keys())!r}.')

    image = load_image(image, mode=None, force_background=None)
    _FN_IMG_SAVE[mimetype](image, dst_file, metadata, **kwargs)
