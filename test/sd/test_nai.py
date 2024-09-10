import pytest
from hbutils.testing import isolated_directory

from imgutils.data import load_image
from imgutils.sd import get_naimeta_from_image, NAIMetaData, add_naimeta_to_image, save_image_with_naimeta
from ..testings import get_testfile


@pytest.fixture()
def nai3_webp_file():
    return get_testfile('nai3_webp.webp')


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
def nai3_clear_rgb_image():
    image = load_image(get_testfile('nai3_clear.png'))
    image.load()
    return image


@pytest.fixture()
def nai3_clear_rgba_image():
    image = load_image(get_testfile('nai3_clear_rgba.png'))
    image.load()
    return image


@pytest.fixture()
def nai3_webp_meta():
    return NAIMetaData(
        software='NovelAI',
        source='Stable Diffusion XL C1E1DE52',
        parameters={
            'prompt': '2girls,side-by-side,nekomata okayu,shiina mahiru,symmetrical pose,general,masterpiece,, '
                      'best quality, amazing quality, very aesthetic, absurdres',
            'steps': 28, 'height': 832, 'width': 1216, 'scale': 5.0, 'uncond_scale': 0.0, 'cfg_rescale': 0.0,
            'seed': 210306140, 'n_samples': 1, 'hide_debug_overlay': False, 'noise_schedule': 'native',
            'legacy_v3_extend': False, 'reference_information_extracted_multiple': [],
            'reference_strength_multiple': [], 'sampler': 'k_euler_ancestral', 'controlnet_strength': 1.0,
            'controlnet_model': None, 'dynamic_thresholding': False, 'dynamic_thresholding_percentile': 0.999,
            'dynamic_thresholding_mimic_scale': 10.0, 'sm': False, 'sm_dyn': False, 'skip_cfg_above_sigma': None,
            'skip_cfg_below_sigma': 0.0, 'lora_unet_weights': None, 'lora_clip_weights': None,
            'deliberate_euler_ancestral_bug': True, 'prefer_brownian': False,
            'cfg_sched_eligibility': 'enable_for_post_summer_samplers', 'explike_fine_detail': False,
            'minimize_sigma_inf': False, 'uncond_per_vibe': True, 'wonky_vibe_correlation': True, 'version': 1,
            'uc': 'lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, '
                  'watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, '
                  'artistic error, username, scan, [abstract], ',
            'request_type': 'PromptGenerateRequest',
            'signed_hash': 'nM6vZLFGJWW7SH2xc4lpRY9sJGbPQKXaUzhUVX/u2NvCAyLg9abn90XBCiNmwqh1hK5hk+o7wYHkPJvhkfAnBg=='
        },
        title=None,
        generation_time=6.494704299024306,
        description='2girls,side-by-side,nekomata okayu,shiina mahiru,symmetrical pose,general,masterpiece,, '
                    'best quality, amazing quality, very aesthetic, absurdres'
    )


@pytest.fixture()
def nai3_meta_without_title():
    return NAIMetaData(
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


@pytest.fixture()
def nai3_meta():
    return NAIMetaData(
        software='NovelAI',
        source='Stable Diffusion XL C1E1DE52',
        title='This is title',
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

    def test_add_naimeta_to_image(self, nai3_clear_rgb_image, nai3_meta_without_title):
        assert get_naimeta_from_image(nai3_clear_rgb_image) is None
        image = add_naimeta_to_image(nai3_clear_rgb_image, metadata=nai3_meta_without_title)
        assert get_naimeta_from_image(image) == pytest.approx(nai3_meta_without_title)

    def test_add_naimeta_to_image_rgba(self, nai3_clear_rgba_image, nai3_meta_without_title):
        assert get_naimeta_from_image(nai3_clear_rgba_image) is None
        image = add_naimeta_to_image(nai3_clear_rgba_image, metadata=nai3_meta_without_title)
        assert get_naimeta_from_image(image) == pytest.approx(nai3_meta_without_title)

    def test_save_image_with_naimeta_both_no(self, nai3_clear_file, nai3_meta_without_title):
        with isolated_directory():
            with pytest.warns(Warning):
                save_image_with_naimeta(
                    nai3_clear_file, 'image.png',
                    metadata=nai3_meta_without_title,
                    save_metainfo=False, add_lsb_meta=False,
                )
            assert get_naimeta_from_image('image.png') is None

    def test_save_image_with_naimeta_with_title(self, nai3_clear_file, nai3_meta):
        with isolated_directory():
            save_image_with_naimeta(nai3_clear_file, 'image.png', metadata=nai3_meta)
            assert get_naimeta_from_image('image.png') == pytest.approx(nai3_meta)

    def test_save_image_with_naimeta_rgba_with_title(self, nai3_clear_rgba_file, nai3_meta):
        with isolated_directory():
            save_image_with_naimeta(nai3_clear_rgba_file, 'image.png', metadata=nai3_meta)
            assert get_naimeta_from_image('image.png') == pytest.approx(nai3_meta)

    def test_save_image_with_naimeta_metainfo_only_with_title(self, nai3_clear_file, nai3_meta):
        with isolated_directory():
            save_image_with_naimeta(nai3_clear_file, 'image.png',
                                    metadata=nai3_meta, add_lsb_meta=False)
            assert get_naimeta_from_image('image.png') == pytest.approx(nai3_meta)

    def test_save_image_with_naimeta_lsbmeta_only_with_title(self, nai3_clear_file, nai3_meta):
        with isolated_directory():
            save_image_with_naimeta(nai3_clear_file, 'image.png',
                                    metadata=nai3_meta, save_metainfo=False)
            assert get_naimeta_from_image('image.png') == pytest.approx(nai3_meta)

    def test_save_image_with_naimeta_both_no_with_title(self, nai3_clear_file, nai3_meta):
        with isolated_directory():
            with pytest.warns(Warning):
                save_image_with_naimeta(
                    nai3_clear_file, 'image.png',
                    metadata=nai3_meta,
                    save_metainfo=False, add_lsb_meta=False,
                )
            assert get_naimeta_from_image('image.png') is None

    @pytest.mark.parametrize(['file'], [
        ('118519492_p0.png',),
        ('118438300_p1.png',),
    ])
    def test_image_error_with_wrong_format(self, file):
        assert get_naimeta_from_image(get_testfile(file)) is None

    def test_get_naimeta_from_image_webp(self, nai3_webp_file, nai3_webp_meta):
        assert get_naimeta_from_image(nai3_webp_file) == pytest.approx(nai3_webp_meta)

    @pytest.mark.parametrize(['ext', 'warns', 'okay'], [
        ('.png', False, True),
        ('.webp', False, True),
        ('.jpg', False, True),
        ('.jpeg', False, True),
        ('.tiff', False, True),
        ('.gif', False, True),
    ])
    def test_save_image_with_naimeta(self, nai3_clear_file, nai3_meta_without_title,
                                     ext, warns, okay):
        with isolated_directory(), pytest.warns(Warning if warns else None):
            save_image_with_naimeta(nai3_clear_file, f'image{ext}', metadata=nai3_meta_without_title)
            assert get_naimeta_from_image(f'image{ext}') == \
                   (pytest.approx(nai3_meta_without_title) if okay else None)

    @pytest.mark.parametrize(['ext', 'warns', 'okay'], [
        ('.png', False, True),
        ('.webp', False, True),
        ('.tiff', False, True),
        ('.gif', False, True),
    ])
    def test_save_image_with_naimeta_rgba(self, nai3_clear_rgba_file, nai3_meta_without_title,
                                          ext, warns, okay):
        with isolated_directory(), pytest.warns(Warning if warns else None):
            save_image_with_naimeta(nai3_clear_rgba_file, f'image{ext}', metadata=nai3_meta_without_title)
            assert get_naimeta_from_image(f'image{ext}') == \
                   (pytest.approx(nai3_meta_without_title) if okay else None)

    @pytest.mark.parametrize(['ext', 'okay'], [
        ('.png', True),
        ('.webp', False),
        ('.jpg', False),
        ('.jpeg', False),
        ('.tiff', True),
        ('.gif', False),
    ])
    def test_save_image_with_naimeta_lsb_true(self, nai3_clear_file, nai3_meta_without_title,
                                              ext, okay):
        with isolated_directory():
            if okay:
                save_image_with_naimeta(nai3_clear_file, f'image{ext}',
                                        add_lsb_meta=True, metadata=nai3_meta_without_title)
                assert get_naimeta_from_image(f'image{ext}') == pytest.approx(nai3_meta_without_title)
            else:
                with pytest.raises(ValueError):
                    save_image_with_naimeta(nai3_clear_file, f'image{ext}',
                                            add_lsb_meta=True, metadata=nai3_meta_without_title)

    @pytest.mark.parametrize(['ext', 'okay'], [
        ('.png', True),
        ('.webp', True),
        ('.jpg', True),
        ('.jpeg', True),
        ('.tiff', False),
        ('.gif', True),
    ])
    def test_save_image_with_naimeta_metainfo_true(self, nai3_clear_file, nai3_meta_without_title,
                                                   ext, okay):
        with isolated_directory():
            if okay:
                save_image_with_naimeta(nai3_clear_file, f'image{ext}',
                                        save_metainfo=True, metadata=nai3_meta_without_title)
                assert get_naimeta_from_image(f'image{ext}') == pytest.approx(nai3_meta_without_title)
            else:
                with pytest.raises(SystemError):
                    save_image_with_naimeta(nai3_clear_file, f'image{ext}',
                                            save_metainfo=True, metadata=nai3_meta_without_title)

    @pytest.mark.parametrize(['ext', 'warns', 'okay'], [
        ('.png', False, True),
        ('.webp', False, True),
        ('.jpg', False, True),
        ('.jpeg', False, True),
        ('.tiff', True, False),
        ('.gif', False, True),
    ])
    def test_save_image_with_naimeta_metainfo_only(self, nai3_clear_file, nai3_meta_without_title,
                                                   ext, warns, okay):
        with isolated_directory(), pytest.warns(Warning if warns else None):
            save_image_with_naimeta(nai3_clear_file, f'image{ext}',
                                    metadata=nai3_meta_without_title, add_lsb_meta=False)
            assert get_naimeta_from_image(f'image{ext}') == \
                   (pytest.approx(nai3_meta_without_title) if okay else None)

    @pytest.mark.parametrize(['ext', 'warns', 'okay'], [
        ('.png', False, True),
        ('.webp', True, False),
        ('.jpg', True, False),
        ('.jpeg', True, False),
        ('.tiff', False, True),
        ('.gif', True, False),
    ])
    def test_save_image_with_naimeta_lsbmeta_only(self, nai3_clear_file, nai3_meta_without_title,
                                                  ext, warns, okay):
        with isolated_directory(), pytest.warns(Warning if warns else None):
            save_image_with_naimeta(nai3_clear_file, f'image{ext}',
                                    metadata=nai3_meta_without_title, save_metainfo=False)
            assert get_naimeta_from_image(f'image{ext}') == \
                   (pytest.approx(nai3_meta_without_title) if okay else None)
