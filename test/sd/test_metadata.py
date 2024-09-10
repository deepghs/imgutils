import os.path
import textwrap
from contextlib import contextmanager

import pytest
from PIL import Image
from hbutils.system import TemporaryDirectory
from hbutils.testing import isolated_directory

from imgutils.sd import get_sdmeta_from_image, SDMetaData, parse_sdmeta_from_text, save_image_with_sdmeta
from test.testings import get_testfile


@pytest.fixture()
def nai3_file():
    return get_testfile('nai3.png')


@pytest.fixture()
def clean_image():
    return get_testfile('nai3_clear.png')


@pytest.fixture()
def sdimg_1():
    yield get_testfile('sd', 'sd1.png')


@pytest.fixture()
def sdimg_1_pil(sdimg_1):
    yield Image.open(sdimg_1)


@pytest.fixture()
def sdimg_1_std():
    return SDMetaData(**{
        'neg_prompt': '',
        'parameters': {
            'CFG scale': 7.0,
            'Denoising strength': 0.75,
            'Model hash': 'e6e8e1fc',
            'Seed': 2468019333,
            'Seed resize from': (-1, -1),
            'Size': (512, 512),
            'Steps': 20
        },
        'prompt': 'a dark photo of the baltimore, phone, selfie, cellphone, '
                  'baltimore_\\(azur_lane\\), blurry, holding_phone, bubble_tea, '
                  'smartphone, choker, 1girl, clothes_around_waist, '
                  'blurry_background, taking_picture, short_hair, v, braid, breasts, '
                  'smile, depth_of_field, necktie, holding, disposable_cup, '
                  'yellow_eyes, shirt, black_choker, collarbone, one_eye_closed, '
                  'short_sleeves, collared_shirt, blurry_foreground, brown_hair, '
                  'bangs, outdoors, hair_between_eyes, white_shirt, large_breasts, '
                  'day, foreshortening, looking_at_viewer, outstretched_arm, cup, '
                  'reaching_out, drinking_straw, ahoge, black_necktie, '
                  'cellphone_picture'
    })


@pytest.fixture()
def sdimg_2():
    yield get_testfile('sd', 'sd2.png')


@pytest.fixture()
def sdimg_2_pil(sdimg_2):
    yield Image.open(sdimg_2)


@pytest.fixture()
def sdimg_2_std():
    return SDMetaData(**{
        'neg_prompt': 'lowres, bad anatomy, bad hands, text, error, missing fingers, '
                      'extra digit, fewer digits, cropped, worst quality, low '
                      'quality, normal quality, jpeg artifacts, signature, watermark, '
                      'username, blurry, bad feet, lowres,bad anatomy,bad '
                      'hands,text,error,missing fingers,extra digit,fewer '
                      'digits,cropped,worst quality,low quality,normal quality,jpeg '
                      'artifacts,signature,watermark,username,blurry,missing '
                      'arms,long neck,Humpbacked, (((futanari))), fat, ((anal)), '
                      '(((anal insertion))), (masturbation),',
        'parameters': {
            'CFG scale': 6.5,
            'Clip skip': 2,
            'Denoising strength': 0.7,
            'ENSD': 31337,
            'Eta': 0.67,
            'First pass size': (0, 0),
            'Model hash': 'a2a802b2',
            'Sampler': 'Euler a',
            'Seed': 1502384588,
            'Size': (512, 768),
            'Steps': 30,
        },
        'prompt': '1girl, 1boy, (sex:1.4), (POV), (an extremely delicate and '
                  'beautiful), ((masterpiece)), illustration, (extremely detailed '
                  'cg), beautiful detailed eyes,best qualtry,(contour '
                  'deepening:1.6),(photorealistic:1.1), (realistic:1), (oil '
                  'painting:1.1), nsfw, masterpiece,best quality, night, intense '
                  'shadows, game_cg, milf, (nude:1.3), long hair, (horse ears:1.2), '
                  'blonde hair, horse tail, yellow eyes, (arknights:1.2), (small '
                  'breast:0.5), small nipples, no_pants, no_bra, (nearl-19000:1.1), '
                  'aqua eyes, (lying:1.25), (arms up), pussy, clitoris, (cum in '
                  'pussy:1.3), cum on body, (seductive smile), (naughty_face), '
                  'looking at viewer, (penis:1.3), (vaginal object insertion:1.3), '
                  'plant, navel, palm tree, long hair, thighs, hetero'
    })


@pytest.fixture()
def sdimg_3():
    yield get_testfile('sd', 'sd3.png')


@pytest.fixture()
def sdimg_3_pil(sdimg_3):
    yield Image.open(sdimg_3)


@pytest.fixture()
def sdimg_3_std():
    return SDMetaData(**{
        'neg_prompt': 'EasyNegative, sketch, duplicate, ugly, huge eyes, text, logo, '
                      'monochrome, worst face, (bad and mutated hands:1.3), (worst '
                      'quality:2.0), (low quality:2.0), (blurry:2.0), horror, '
                      'geometry, bad_prompt, (bad hands), (missing fingers), multiple '
                      'limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, '
                      '(extra digit and hands and fingers and legs and arms:1.4), '
                      '((2girl)), (deformed fingers:1.2), (long '
                      'fingers:1.2),(bad-artist-anime), bad-artist, bad hand, blush, '
                      '(lipstick),skindentation, tie, ((big_breast)), (nipple), '
                      'thighhighs, pubic_hair, pussy, black and white,(3d), '
                      '((realistic)),blurry,nipple slip, (nipple), blush, '
                      'head_out_of_frame,curvy,',
        'parameters': {
            'CFG scale': 7,
            'Clip skip': 2,
            'Model': 'AniDosMix',
            'Model hash': 'eb49192009',
            'Sampler': 'DDIM',
            'Seed': 3827064803,
            'Size': (512, 848),
            'Steps': 20
        },
        'prompt': '(extremely delicate and beautiful), best quality, official art, '
                  'global illumination, soft shadow, super detailed, Japanese light '
                  'novel cover, 4K, metal_texture, (striped_background), super '
                  'detailed background, more detailed, rich detailed, extremely '
                  'detailed CG unity 8k wallpaper, ((unreal)), '
                  'sci-fi,(fantasy),(masterpiece),(super delicate), (illustration), '
                  '(extremely delicate and beautiful), anime coloring,\n'
                  '(silver_skin), ((high-cut silver_impossible_bodysuit), '
                  '((gem_on_chest)),(high-cut_silver_mechanical_leotard)),headgear,\n'
                  '(focus-on:1.1),(1_girl),((solo)),slim_waist,white hair, long hair, '
                  'luminous yellow eyes,(medium_breast:1.2), '
                  '(Indistinct_cameltoe:0.9), (flat_crotch:1.1),(coquettish), '
                  '(squint:1.4),(evil_smile :1.35),(dark_persona), [open mouth: 0.7], '
                  'standing,[wet:0.7],\n'
                  'slim_face, tall_girl,(mature),mature_face, (slim_figure), '
                  '(slim_legs:1.1), (groin:1.1), ((bare_thighs)),'
    })


@pytest.fixture()
def sdimg_4():
    yield get_testfile('sd', 'sd4.png')


@pytest.fixture()
def sdimg_4_pil(sdimg_4):
    yield Image.open(sdimg_4)


@pytest.fixture()
def sdimg_4_std():
    return SDMetaData(**{
        'neg_prompt': 'Neg1,Negative,',
        'parameters': {
            'CFG scale': 7,
            'ControlNet 0': 'preprocessor: openpose, model: '
                            'control_v11p_sd15_openpose [cab727d4], '
                            'weight: 1, starting/ending: (0, 1), resize '
                            'mode: Crop and Resize, pixel perfect: False, '
                            'control mode: Balanced, preprocessor params: '
                            '(512, 64, 64)',
            'Denoising strength': 0.7,
            'Hires upscale': 2,
            'Hires upscaler': 'Latent',
            'Model': 'CuteMix',
            'Model hash': '72bd94132e',
            'Sampler': 'DPM++ 2M SDE Karras',
            'Seed': 2647703743,
            'Size': (768, 768),
            'Steps': 20,
            'TI hashes': 'Neg1: 339cc9210f70, Negative: 66a7279a88dd',
            'Version': 'v1.5.1'
        },
        'prompt': '1girl, solo, blue eyes, black footwear, white hair, looking at '
                  'viewer, shoes, full body, standing, bangs, indoors, wide sleeves, '
                  'ahoge, dress, closed mouth, blush, long sleeves, potted plant, '
                  'bag, plant, hair bun, '
                  'window,<lora:BlueArchive10:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1>,BlueArchive,'
    })


@pytest.fixture()
def sdimg_none():
    yield get_testfile('genshin_post.jpg')


@pytest.fixture()
def sdimg_none_pil(sdimg_none):
    yield Image.open(sdimg_none)


@pytest.fixture()
def sdtext_simple(sdimg_2_pil):
    yield sdimg_2_pil.info['parameters']


@pytest.fixture()
def sdtext_simple_std(sdimg_2_std):
    yield sdimg_2_std


@pytest.fixture()
def sdtext_complex(sdimg_4_pil):
    yield sdimg_4_pil.info['parameters']


@pytest.fixture()
def sdtext_complex_std(sdimg_4_std):
    yield sdimg_4_std


@pytest.fixture()
def sdtext_no_negative():
    return textwrap.dedent("""
1girl, solo, blue eyes, black footwear, white hair, looking at viewer, shoes, full body, standing, bangs, indoors, wide sleeves, ahoge, dress, closed mouth, blush, long sleeves, potted plant, bag, plant, hair bun, window,<lora:BlueArchive10:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1>,BlueArchive,
Steps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 7, Seed: 2647703743, Size: 768x768, Model hash: 72bd94132e, Model: CuteMix, Denoising strength: 0.7, ControlNet 0: "preprocessor: openpose, model: control_v11p_sd15_openpose [cab727d4], weight: 1, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: False, control mode: Balanced, preprocessor params: (512, 64, 64)", Hires upscale: 2, Hires upscaler: Latent, TI hashes: "Neg1: 339cc9210f70, Negative: 66a7279a88dd", Version: v1.5.1
    """).strip()


@pytest.fixture()
def sdtext_no_negative_std():
    return SDMetaData(**{
        'neg_prompt': '',
        'parameters': {
            'CFG scale': 7,
            'ControlNet 0': 'preprocessor: openpose, model: '
                            'control_v11p_sd15_openpose [cab727d4], '
                            'weight: 1, starting/ending: (0, 1), resize '
                            'mode: Crop and Resize, pixel perfect: False, '
                            'control mode: Balanced, preprocessor params: '
                            '(512, 64, 64)',
            'Denoising strength': 0.7,
            'Hires upscale': 2,
            'Hires upscaler': 'Latent',
            'Model': 'CuteMix',
            'Model hash': '72bd94132e',
            'Sampler': 'DPM++ 2M SDE Karras',
            'Seed': 2647703743,
            'Size': (768, 768),
            'Steps': 20,
            'TI hashes': 'Neg1: 339cc9210f70, Negative: 66a7279a88dd',
            'Version': 'v1.5.1'
        },
        'prompt': '1girl, solo, blue eyes, black footwear, white hair, looking at '
                  'viewer, shoes, full body, standing, bangs, indoors, wide sleeves, '
                  'ahoge, dress, closed mouth, blush, long sleeves, potted plant, '
                  'bag, plant, hair bun, '
                  'window,<lora:BlueArchive10:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1>,BlueArchive,'
    })


@pytest.fixture()
def sdtext_no_params():
    return textwrap.dedent("""
1girl, solo, blue eyes, black footwear, white hair, looking at viewer, shoes, full body, standing, bangs, indoors, wide sleeves, ahoge, dress, closed mouth, blush, long sleeves, potted plant, bag, plant, hair bun, window,<lora:BlueArchive10:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1>,BlueArchive,
Negative prompt: Neg1,Negative,
    """).strip()


@pytest.fixture()
def sdtext_no_params_std():
    return SDMetaData(**{
        'neg_prompt': 'Neg1,Negative,',
        'parameters': {},
        'prompt': '1girl, solo, blue eyes, black footwear, white hair, looking at '
                  'viewer, shoes, full body, standing, bangs, indoors, wide sleeves, '
                  'ahoge, dress, closed mouth, blush, long sleeves, potted plant, '
                  'bag, plant, hair bun, '
                  'window,<lora:BlueArchive10:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1>,BlueArchive,'
    })


@pytest.mark.unittest
class TestSdMetadata:
    def test_get_sdmeta_from_image(self, sdimg_1, sdimg_2, sdimg_3, sdimg_4,
                                   sdimg_1_pil, sdimg_2_pil, sdimg_3_pil, sdimg_4_pil,
                                   sdimg_1_std, sdimg_2_std, sdimg_3_std, sdimg_4_std):
        assert get_sdmeta_from_image(sdimg_1) == sdimg_1_std
        assert get_sdmeta_from_image(sdimg_2) == sdimg_2_std
        assert get_sdmeta_from_image(sdimg_3) == sdimg_3_std
        assert get_sdmeta_from_image(sdimg_4) == sdimg_4_std

        assert get_sdmeta_from_image(sdimg_1_pil) == sdimg_1_std
        assert get_sdmeta_from_image(sdimg_2_pil) == sdimg_2_std
        assert get_sdmeta_from_image(sdimg_3_pil) == sdimg_3_std
        assert get_sdmeta_from_image(sdimg_4_pil) == sdimg_4_std

        assert get_sdmeta_from_image(sdimg_1) != sdimg_2_std
        assert get_sdmeta_from_image(sdimg_1) != sdimg_3_std
        assert get_sdmeta_from_image(sdimg_1) != sdimg_4_std
        assert get_sdmeta_from_image(sdimg_2) != sdimg_1_std
        assert get_sdmeta_from_image(sdimg_2) != sdimg_3_std
        assert get_sdmeta_from_image(sdimg_2) != sdimg_4_std
        assert get_sdmeta_from_image(sdimg_3) != sdimg_1_std
        assert get_sdmeta_from_image(sdimg_3) != sdimg_2_std
        assert get_sdmeta_from_image(sdimg_3) != sdimg_4_std
        assert get_sdmeta_from_image(sdimg_4) != sdimg_1_std
        assert get_sdmeta_from_image(sdimg_4) != sdimg_2_std
        assert get_sdmeta_from_image(sdimg_4) != sdimg_3_std

        assert get_sdmeta_from_image(sdimg_1_pil) != sdimg_2_std
        assert get_sdmeta_from_image(sdimg_1_pil) != sdimg_3_std
        assert get_sdmeta_from_image(sdimg_1_pil) != sdimg_4_std
        assert get_sdmeta_from_image(sdimg_2_pil) != sdimg_1_std
        assert get_sdmeta_from_image(sdimg_2_pil) != sdimg_3_std
        assert get_sdmeta_from_image(sdimg_2_pil) != sdimg_4_std
        assert get_sdmeta_from_image(sdimg_3_pil) != sdimg_1_std
        assert get_sdmeta_from_image(sdimg_3_pil) != sdimg_2_std
        assert get_sdmeta_from_image(sdimg_3_pil) != sdimg_4_std
        assert get_sdmeta_from_image(sdimg_4_pil) != sdimg_1_std
        assert get_sdmeta_from_image(sdimg_4_pil) != sdimg_2_std
        assert get_sdmeta_from_image(sdimg_4_pil) != sdimg_3_std

    def test_get_sdmeta_from_image_none(self, sdimg_none, sdimg_none_pil):
        assert get_sdmeta_from_image(sdimg_none) is None
        assert get_sdmeta_from_image(sdimg_none_pil) is None

    def test_parse_sdmeta_from_text(self, sdtext_simple, sdtext_complex, sdtext_no_negative, sdtext_no_params,
                                    sdtext_simple_std, sdtext_complex_std, sdtext_no_negative_std,
                                    sdtext_no_params_std):
        assert parse_sdmeta_from_text(sdtext_simple) == sdtext_simple_std
        assert parse_sdmeta_from_text(sdtext_complex) == sdtext_complex_std
        assert parse_sdmeta_from_text(sdtext_no_negative) == sdtext_no_negative_std
        assert parse_sdmeta_from_text(sdtext_no_params) == sdtext_no_params_std

    def test_sdmeta_str(self, text_aligner, sdimg_1_std, sdimg_2_std, sdimg_3_std, sdimg_4_std,
                        sdimg_1_pil, sdimg_2_pil):
        text_aligner.assert_equal(
            str(sdimg_1_std),
            r"""
a dark photo of the baltimore, phone, selfie, cellphone, baltimore_\(azur_lane\), blurry, holding_phone, bubble_tea, smartphone, choker, 1girl, clothes_around_waist, blurry_background, taking_picture, short_hair, v, braid, breasts, smile, depth_of_field, necktie, holding, disposable_cup, yellow_eyes, shirt, black_choker, collarbone, one_eye_closed, short_sleeves, collared_shirt, blurry_foreground, brown_hair, bangs, outdoors, hair_between_eyes, white_shirt, large_breasts, day, foreshortening, looking_at_viewer, outstretched_arm, cup, reaching_out, drinking_straw, ahoge, black_necktie, cellphone_picture                                                                                                                                                 
Steps: 20, CFG scale: 7.0, Seed: 2468019333, Size: 512x512, Model hash: e6e8e1fc, Seed resize from: -1x-1, Denoising strength: 0.75
            """
        )
        assert parse_sdmeta_from_text(str(sdimg_1_std)) == sdimg_1_std

        text_aligner.assert_equal(
            str(sdimg_2_std),
            """
1girl, 1boy, (sex:1.4), (POV), (an extremely delicate and beautiful), ((masterpiece)), illustration, (extremely detailed cg), beautiful detailed eyes,best qualtry,(contour deepening:1.6),(photorealistic:1.1), (realistic:1), (oil painting:1.1), nsfw, masterpiece,best quality, night, intense shadows, game_cg, milf, (nude:1.3), long hair, (horse ears:1.2), blonde hair, horse tail, yellow eyes, (arknights:1.2), (small breast:0.5), small nipples, no_pants, no_bra, (nearl-19000:1.1), aqua eyes, (lying:1.25), (arms up), pussy, clitoris, (cum in pussy:1.3), cum on body, (seductive smile), (naughty_face), looking at viewer, (penis:1.3), (vaginal object insertion:1.3), plant, navel, palm tree, long hair, thighs, hetero
Negative prompt: lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, lowres,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,missing arms,long neck,Humpbacked, (((futanari))), fat, ((anal)), (((anal insertion))), (masturbation),
Steps: 30, Sampler: Euler a, CFG scale: 6.5, Seed: 1502384588, Size: 512x768, Model hash: a2a802b2, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, Eta: 0.67, First pass size: 0x0
            """
        )
        assert parse_sdmeta_from_text(str(sdimg_2_std)) == sdimg_2_std

        text_aligner.assert_equal(
            str(sdimg_3_std),
            """
(extremely delicate and beautiful), best quality, official art, global illumination, soft shadow, super detailed, Japanese light novel cover, 4K, metal_texture, (striped_background), super detailed background, more detailed, rich detailed, extremely detailed CG unity 8k wallpaper, ((unreal)), sci-fi,(fantasy),(masterpiece),(super delicate), (illustration), (extremely delicate and beautiful), anime coloring,
(silver_skin), ((high-cut silver_impossible_bodysuit), ((gem_on_chest)),(high-cut_silver_mechanical_leotard)),headgear,
(focus-on:1.1),(1_girl),((solo)),slim_waist,white hair, long hair, luminous yellow eyes,(medium_breast:1.2), (Indistinct_cameltoe:0.9), (flat_crotch:1.1),(coquettish), (squint:1.4),(evil_smile :1.35),(dark_persona), [open mouth: 0.7], standing,[wet:0.7],
slim_face, tall_girl,(mature),mature_face, (slim_figure), (slim_legs:1.1), (groin:1.1), ((bare_thighs)),
Negative prompt: EasyNegative, sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), ((2girl)), (deformed fingers:1.2), (long fingers:1.2),(bad-artist-anime), bad-artist, bad hand, blush, (lipstick),skindentation, tie, ((big_breast)), (nipple), thighhighs, pubic_hair, pussy, black and white,(3d), ((realistic)),blurry,nipple slip, (nipple), blush, head_out_of_frame,curvy,
Steps: 20, Sampler: DDIM, CFG scale: 7, Seed: 3827064803, Size: 512x848, Model hash: eb49192009, Model: AniDosMix, Clip skip: 2
            """
        )
        assert parse_sdmeta_from_text(str(sdimg_3_std)) == sdimg_3_std

        text_aligner.assert_equal(
            str(sdimg_4_std),
            """
1girl, solo, blue eyes, black footwear, white hair, looking at viewer, shoes, full body, standing, bangs, indoors, wide sleeves, ahoge, dress, closed mouth, blush, long sleeves, potted plant, bag, plant, hair bun, window,<lora:BlueArchive10:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1>,BlueArchive,
Negative prompt: Neg1,Negative,
Steps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 7, Seed: 2647703743, Size: 768x768, Model hash: 72bd94132e, Model: CuteMix, Denoising strength: 0.7, ControlNet 0: "preprocessor: openpose, model: control_v11p_sd15_openpose [cab727d4], weight: 1, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: False, control mode: Balanced, preprocessor params: (512, 64, 64)", Hires upscale: 2, Hires upscaler: Latent, TI hashes: "Neg1: 339cc9210f70, Negative: 66a7279a88dd", Version: v1.5.1
            """
        )
        assert parse_sdmeta_from_text(str(sdimg_4_std)) == sdimg_4_std

    def test_pnginfo(self, sdimg_1_std, sdimg_2_std, sdimg_3_std, sdimg_4_std):
        @contextmanager
        def _test_pnginfo(pinfo):
            with TemporaryDirectory() as td:
                img_file = os.path.join(td, 'img.png')
                Image.new('RGB', (256, 256), 'white').save(img_file, pnginfo=pinfo)
                yield img_file

        with _test_pnginfo(sdimg_1_std.pnginfo) as f:
            assert get_sdmeta_from_image(f) == sdimg_1_std
        with _test_pnginfo(sdimg_2_std.pnginfo) as f:
            assert get_sdmeta_from_image(f) == sdimg_2_std
        with _test_pnginfo(sdimg_3_std.pnginfo) as f:
            assert get_sdmeta_from_image(f) == sdimg_3_std
        with _test_pnginfo(sdimg_4_std.pnginfo) as f:
            assert get_sdmeta_from_image(f) == sdimg_4_std

    def test_empty_info_parse(self):
        assert parse_sdmeta_from_text('') == SDMetaData('', '', {})

    @pytest.mark.parametrize(['ext', 'result'], [
        ('.png', True),
        ('.jpg', True),
        ('.jpeg', True),
        ('.webp', True),
        ('.gif', True),
        ('.tiff', SystemError),
    ])
    def test_save_image_with_sdmeta(self, clean_image, sdimg_4_std, ext, result):
        assert get_sdmeta_from_image(clean_image) is None
        with isolated_directory():
            if isinstance(result, type) and issubclass(result, Exception):
                with pytest.raises(result):
                    save_image_with_sdmeta(clean_image, f'image{ext}', metadata=sdimg_4_std)
            else:
                save_image_with_sdmeta(clean_image, f'image{ext}', metadata=sdimg_4_std)
                assert get_sdmeta_from_image(f'image{ext}') == sdimg_4_std

    @pytest.mark.parametrize(['file'], [
        ('nai3.png',),
        ('nai3_clear.png',),
        ('nai3_info_rgb.png',),
    ])
    def test_clean_image(self, file):
        assert get_sdmeta_from_image(get_testfile(file)) is None
