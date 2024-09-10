import pytest
from PIL import Image
from hbutils.testing import tmatrix, isolated_directory

from imgutils.metadata import read_geninfo_parameters, read_geninfo_exif, read_geninfo_gif, write_geninfo_parameters, \
    write_geninfo_exif, write_geninfo_gif
from test.testings import get_testfile


@pytest.fixture
def a41_webui_file():
    return get_testfile('nude_girl.png')


@pytest.fixture()
def exif_file():
    return get_testfile('nai3_webp.webp')


@pytest.fixture()
def gif_file():
    return get_testfile('nian_geninfo_gif.gif')


@pytest.fixture()
def png_rgb_clean():
    return get_testfile('nai3_clear.png')


@pytest.fixture()
def png_comment_str():
    return get_testfile('nai3_clear_comment_str.png')


@pytest.mark.unittest
class TestMetadataGeninfo:
    def test_read_geninfo_parameters(self, a41_webui_file):
        assert read_geninfo_parameters(a41_webui_file) == (
            'nsfw, masterpiece,best quality, game_cg, 1girl, solo, (((nude))), red hair, '
            'long hair, purple eyes, horn, arknights, (surtr), medium breast, small '
            'nipples, (((lying))), aqua eyes, (armpits), (spread legs), pussy/vaginal, '
            'clitoris, ((pussy_juice)), naughty_face, ((endured_face)), looking at '
            'viewer, (((caught))), walk-in,\n'
            'Negative prompt: lowres, bad anatomy, bad hands, text, error, missing '
            'fingers, extra digit, fewer digits, cropped, worst quality, low quality, '
            'normal quality, jpeg artifacts, signature, watermark, username, blurry, bad '
            'feet, lowres,bad anatomy,bad hands,text,error,missing fingers,extra '
            'digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg '
            'artifacts,signature,watermark,username,blurry,missing arms,long '
            'neck,Humpbacked, {{{futanari}}}, fat, anal, anal insertion\n'
            'Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 3641508112, Size: 512x768, '
            'Model hash: e6e8e1fc, Denoising strength: 0.75, Mask blur: 4')

    @pytest.mark.parametrize(['file'], [
        ('6125901.jpg',),
        ('nian.png',),
        ('nai3_info_rgb.png',),
        ('nai3_webp.webp',),
        ('nian_geninfo_gif.gif',),
    ])
    def test_read_geninfo_parameters_empty(self, file):
        assert not read_geninfo_parameters(get_testfile(file))

    def test_read_geninfo_exif(self, exif_file):
        assert read_geninfo_exif(exif_file) == (
            '{\n'
            '    "Title": "AI generated image",\n'
            '    "Description": "2girls,side-by-side,nekomata okayu,shiina '
            'mahiru,symmetrical pose,general,masterpiece,, best quality, amazing quality, '
            'very aesthetic, absurdres",\n'
            '    "Software": "NovelAI",\n'
            '    "Source": "Stable Diffusion XL C1E1DE52",\n'
            '    "Generation time": "6.494704299024306",\n'
            '    "Comment": {\n'
            '        "prompt": "2girls,side-by-side,nekomata okayu,shiina '
            'mahiru,symmetrical pose,general,masterpiece,, best quality, amazing quality, '
            'very aesthetic, absurdres",\n'
            '        "steps": 28,\n'
            '        "height": 832,\n'
            '        "width": 1216,\n'
            '        "scale": 5.0,\n'
            '        "uncond_scale": 0.0,\n'
            '        "cfg_rescale": 0.0,\n'
            '        "seed": 210306140,\n'
            '        "n_samples": 1,\n'
            '        "hide_debug_overlay": false,\n'
            '        "noise_schedule": "native",\n'
            '        "legacy_v3_extend": false,\n'
            '        "reference_information_extracted_multiple": [],\n'
            '        "reference_strength_multiple": [],\n'
            '        "sampler": "k_euler_ancestral",\n'
            '        "controlnet_strength": 1.0,\n'
            '        "controlnet_model": null,\n'
            '        "dynamic_thresholding": false,\n'
            '        "dynamic_thresholding_percentile": 0.999,\n'
            '        "dynamic_thresholding_mimic_scale": 10.0,\n'
            '        "sm": false,\n'
            '        "sm_dyn": false,\n'
            '        "skip_cfg_above_sigma": null,\n'
            '        "skip_cfg_below_sigma": 0.0,\n'
            '        "lora_unet_weights": null,\n'
            '        "lora_clip_weights": null,\n'
            '        "deliberate_euler_ancestral_bug": true,\n'
            '        "prefer_brownian": false,\n'
            '        "cfg_sched_eligibility": "enable_for_post_summer_samplers",\n'
            '        "explike_fine_detail": false,\n'
            '        "minimize_sigma_inf": false,\n'
            '        "uncond_per_vibe": true,\n'
            '        "wonky_vibe_correlation": true,\n'
            '        "version": 1,\n'
            '        "uc": "lowres, {bad}, error, fewer, extra, missing, worst quality, '
            'jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic '
            'aberration, signature, extra digits, artistic error, username, scan, '
            '[abstract], ",\n'
            '        "request_type": "PromptGenerateRequest",\n'
            '        "signed_hash": '
            '"nM6vZLFGJWW7SH2xc4lpRY9sJGbPQKXaUzhUVX/u2NvCAyLg9abn90XBCiNmwqh1hK5hk+o7wYHkPJvhkfAnBg=="\n'
            '    }\n'
            '}'
        )

    @pytest.mark.parametrize(['file'], [
        ('nude_girl.png',),
        ('6125901.jpg',),
        ('nian.png',),
        ('nai3_info_rgb.png',),
        ('nian_geninfo_gif.gif',),
    ])
    def test_read_geninfo_exif_empty(self, file):
        assert not read_geninfo_exif(get_testfile(file))

    def test_read_geninfo_gif(self, gif_file):
        assert read_geninfo_gif(gif_file) == 'this is nian dragon dragon, lmao'

    @pytest.mark.parametrize(['file'], [
        ('nai3_webp.webp',),
        ('nude_girl.png',),
        ('6125901.jpg',),
        ('nian.png',),
        ('nai3_info_rgb.png',),
    ])
    def test_read_geninfo_gif_empty(self, file):
        assert not read_geninfo_gif(get_testfile(file))

    @pytest.mark.parametrize(*tmatrix({
        'ext': ['.png'],
        'text': ['', 'this is nian dragon dragon, lmao', 'qazwsxedcrfvtgbyhn' * 100],
    }, mode='matrix'))
    def test_write_geninfo_parameters(self, ext, text, png_rgb_clean):
        assert not read_geninfo_parameters(png_rgb_clean)
        with isolated_directory():
            write_geninfo_parameters(png_rgb_clean, f'image{ext}', geninfo=text)
            assert read_geninfo_parameters(f'image{ext}') == text

    @pytest.mark.parametrize(*tmatrix({
        'ext': ['.webp', '.jpg', '.jpeg', '.png'],
        'text': ['', 'this is nian dragon dragon, lmao', 'qazwsxedcrfvtgbyhn' * 100],
    }, mode='matrix'))
    def test_write_geninfo_exif(self, ext, text, png_rgb_clean):
        assert not read_geninfo_exif(png_rgb_clean)
        with isolated_directory():
            write_geninfo_exif(png_rgb_clean, f'image{ext}', geninfo=text)
            assert read_geninfo_exif(f'image{ext}') == text

    @pytest.mark.parametrize(*tmatrix({
        'ext': ['.gif'],
        'text': ['', 'this is nian dragon dragon, lmao', 'qazwsxedcrfvtgbyhn' * 100],
    }, mode='matrix'))
    def test_write_geninfo_gif(self, ext, text, png_rgb_clean):
        assert not read_geninfo_gif(png_rgb_clean)
        with isolated_directory():
            write_geninfo_gif(png_rgb_clean, f'image{ext}', geninfo=text)
            assert read_geninfo_gif(f'image{ext}') == (text if text else None)

    def test_read_geninfo_gif_comment_str(self, png_comment_str):
        image = Image.open(png_comment_str)
        assert isinstance(image.info.get('comment'), str)
        assert read_geninfo_gif(png_comment_str) is None
        assert read_geninfo_gif(image) is None
