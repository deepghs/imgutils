import io
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Dict, Any, Optional

from PIL.PngImagePlugin import PngInfo

from ..data import ImageTyping, load_image

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
    prompt: str
    neg_prompt: str
    parameters: Dict[str, Any]

    def __str__(self):
        return self._sdmeta_text()

    def _sdmeta_text(self):
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
    def pnginfo(self) -> PngInfo:
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


def parse_sdmeta_from_text(x: str):
    x = textwrap.dedent(x).strip()
    *prompt_lines, argument_line = x.splitlines(keepends=False)
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


def get_sdmeta_from_image(image: ImageTyping) -> Optional[SDMetaData]:
    image = load_image(image, mode=None, force_background=None)
    pnginfo_text = image.info.get('parameters')
    if pnginfo_text:
        return parse_sdmeta_from_text(pnginfo_text)
    else:
        return None
