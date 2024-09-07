import pytest

from imgutils.sd import get_naimeta_from_image, NAIMetadata
from ..testings import get_testfile


@pytest.fixture()
def nai3_file():
    return get_testfile('nai3.png')


@pytest.fixture()
def nai3_info_rgb_file():
    return get_testfile('nai3_info_rgb.png')


@pytest.fixture()
def nai3_clear_file():
    return get_testfile('nai3_clear.png')


@pytest.fixture()
def nai3_clear_rgba_file():
    return get_testfile('nai3_clear_rgba.png')


@pytest.fixture()
def nai3_meta_without_title():
    return NAIMetadata(
        software='NovelAI',
        source='Stable Diffusion XL C1E1DE52',
        title=None,
        generation_time=7.801080600824207,
        description='mudrock (arknights),nsfw,nude,nipple,pussy,1girl, {{small breast}}, artist:as109, '
                    'artist:sigm@,  [artist:neko_(yanshoujie), artist:marumoru, artist:yoneyama_mai], '
                    'year 2023, solo, midriff, thigh cutout,  hair ornament, looking at viewer, navel,  '
                    'floating ink brush background, jewelry, collarbone, tassel, bangs, talisman, cowboy shot, '
                    'gold stripes, floating ink around, high contrast background, beautiful detailed eyes,',
        parameters={
            'prompt': 'mudrock (arknights),nsfw,nude,nipple,pussy,1girl, {{small breast}}, artist:as109, '
                      'artist:sigm@,  [artist:neko_(yanshoujie), artist:marumoru, artist:yoneyama_mai], '
                      'year 2023, solo, midriff, thigh cutout,  hair ornament, looking at viewer, navel,  '
                      'floating ink brush background, jewelry, collarbone, tassel, bangs, talisman, cowboy shot, '
                      'gold stripes, floating ink around, high contrast background, beautiful detailed eyes,',
            'steps': 28, 'height': 1216, 'width': 832, 'scale': 5.0, 'uncond_scale': 0.0, 'cfg_rescale': 0.0,
            'seed': 1016281108, 'n_samples': 1, 'hide_debug_overlay': False, 'noise_schedule': 'karras',
            'legacy_v3_extend': False, 'reference_information_extracted_multiple': [],
            'reference_strength_multiple': [], 'sampler': 'k_dpmpp_2s_ancestral', 'controlnet_strength': 1.0,
            'controlnet_model': None, 'dynamic_thresholding': False, 'dynamic_thresholding_percentile': 0.999,
            'dynamic_thresholding_mimic_scale': 10.0, 'sm': True, 'sm_dyn': False, 'skip_cfg_above_sigma': None,
            'skip_cfg_below_sigma': 0.0, 'lora_unet_weights': None, 'lora_clip_weights': None,
            'deliberate_euler_ancestral_bug': True, 'prefer_brownian': False,
            'cfg_sched_eligibility': 'enable_for_post_summer_samplers', 'explike_fine_detail': False,
            'minimize_sigma_inf': False, 'uncond_per_vibe': True, 'wonky_vibe_correlation': True, 'version': 1,
            'uc': 'lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, '
                  'watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, '
                  'artistic error, username, scan, [abstract], loli,  blush,  man, skindentation,ribs, '
                  'pubic hair, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, '
                  'fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, '
                  'signature, watermark, username',
            'request_type': 'PromptGenerateRequest',
            'signed_hash': 'BdmflmfBoxELpIxO2FP7WiRNM08uqcRDeO0HcWbHReQrP8UZq4LkZkaV09BpsXY3UfyGJ1tSX1JRyCedJFu3CQ=='
        })


@pytest.mark.unittest
class TestSDNai:
    def test_get_naimeta_from_image(self, nai3_file, nai3_meta_without_title):
        assert get_naimeta_from_image(nai3_file) == pytest.approx(nai3_meta_without_title)

    def test_get_naimeta_from_image_info_rgb(self, nai3_info_rgb_file, nai3_meta_without_title):
        assert get_naimeta_from_image(nai3_info_rgb_file) == pytest.approx(nai3_meta_without_title)

    def test_get_naimeta_from_image_cleared_rgb(self, nai3_clear_file, nai3_meta_without_title):
        assert get_naimeta_from_image(nai3_clear_file) is None

    def test_get_naimeta_from_image_cleared_rgba(self, nai3_clear_rgba_file, nai3_meta_without_title):
        assert get_naimeta_from_image(nai3_clear_rgba_file) is None
