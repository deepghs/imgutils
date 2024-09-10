import pytest
from PIL.PngImagePlugin import PngInfo
from hbutils.testing import isolated_directory

from imgutils.metadata import read_lsb_metadata, LSBReadError, read_lsb_raw_bytes, write_lsb_raw_bytes, \
    write_lsb_metadata
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
def metadata_expected():
    return {
        'Comment': '{"prompt": "mudrock (arknights),nsfw,nude,nipple,pussy,1girl, '
                   '{{small breast}}, artist:as109, artist:sigm@,  '
                   '[artist:neko_(yanshoujie), artist:marumoru, artist:yoneyama_mai], '
                   'year 2023, solo, midriff, thigh cutout,  hair ornament, looking '
                   'at viewer, navel,  floating ink brush background, jewelry, '
                   'collarbone, tassel, bangs, talisman, cowboy shot, gold stripes, '
                   'floating ink around, high contrast background, beautiful detailed '
                   'eyes,", "steps": 28, "height": 1216, "width": 832, "scale": 5.0, '
                   '"uncond_scale": 0.0, "cfg_rescale": 0.0, "seed": 1016281108, '
                   '"n_samples": 1, "hide_debug_overlay": false, "noise_schedule": '
                   '"karras", "legacy_v3_extend": false, '
                   '"reference_information_extracted_multiple": [], '
                   '"reference_strength_multiple": [], "sampler": '
                   '"k_dpmpp_2s_ancestral", "controlnet_strength": 1.0, '
                   '"controlnet_model": null, "dynamic_thresholding": false, '
                   '"dynamic_thresholding_percentile": 0.999, '
                   '"dynamic_thresholding_mimic_scale": 10.0, "sm": true, "sm_dyn": '
                   'false, "skip_cfg_above_sigma": null, "skip_cfg_below_sigma": 0.0, '
                   '"lora_unet_weights": null, "lora_clip_weights": null, '
                   '"deliberate_euler_ancestral_bug": true, "prefer_brownian": false, '
                   '"cfg_sched_eligibility": "enable_for_post_summer_samplers", '
                   '"explike_fine_detail": false, "minimize_sigma_inf": false, '
                   '"uncond_per_vibe": true, "wonky_vibe_correlation": true, '
                   '"version": 1, "uc": "lowres, {bad}, error, fewer, extra, missing, '
                   'worst quality, jpeg artifacts, bad quality, watermark, '
                   'unfinished, displeasing, chromatic aberration, signature, extra '
                   'digits, artistic error, username, scan, [abstract], loli,  '
                   'blush,  man, skindentation,ribs, pubic hair, lowres, bad anatomy, '
                   'bad hands, text, error, missing fingers, extra digit, fewer '
                   'digits, cropped, worst quality, low quality, normal quality, jpeg '
                   'artifacts, signature, watermark, username", "request_type": '
                   '"PromptGenerateRequest", "signed_hash": '
                   '"BdmflmfBoxELpIxO2FP7WiRNM08uqcRDeO0HcWbHReQrP8UZq4LkZkaV09BpsXY3UfyGJ1tSX1JRyCedJFu3CQ=="}',
        'Description': 'mudrock (arknights),nsfw,nude,nipple,pussy,1girl, {{small '
                       'breast}}, artist:as109, artist:sigm@,  '
                       '[artist:neko_(yanshoujie), artist:marumoru, '
                       'artist:yoneyama_mai], year 2023, solo, midriff, thigh '
                       'cutout,  hair ornament, looking at viewer, navel,  floating '
                       'ink brush background, jewelry, collarbone, tassel, bangs, '
                       'talisman, cowboy shot, gold stripes, floating ink around, '
                       'high contrast background, beautiful detailed eyes,',
        'Generation time': '7.801080600824207',
        'Software': 'NovelAI',
        'Source': 'Stable Diffusion XL C1E1DE52'
    }


@pytest.fixture()
def pnginfo_expected(metadata_expected):
    pnginfo = PngInfo()
    for key, value in metadata_expected.items():
        pnginfo.add_text(key, value)
    return pnginfo


@pytest.fixture()
def raw_bytes_expected():
    return bytearray(b'\x1f\x8b\x08\x00\xbcE\xd4f\x02\xff\xedU\xdbn\xdc6\x10\xfd\x15b'
                     b'\x9f\x12@\x08\xa4ub\xaf\x03\x04h}\xc9\xc5p\x13\xc7F\x1a7'
                     b'^\x83\x18\x89#\x89\x11E\xca\xbcx\xad\x18\xfe\xf7\x0e%e\xbd\x1b'
                     b'8\x1fP\xa0O\xc2p.\x9c9sxt?;BWX\xd9yi\xf4\xec5\x9b\xb5AXS4\xec\x19'
                     b"\xd8F\xcb\xaa\xf6\xeey\xa2]\xb9Jt\x10\x98h\xd9u\n\x93.8\xd7'Y"
                     b'%\xadJ\xd8\xfd\xbdkA)\x96[\x04\xe7\x1f\x1e\x12\x06\xd6K\xe7'
                     b'_\x83\xcb\xd2\xfd\xb5\xe5d\xd5\xfe\x910v5\xd9\x1a\x1b\xc3\x9f\xf5'
                     b'\xa0]m\xc2w\x89\xcf\xd7\x91-\xd8\xd0\x1a\x1b\xd6\x07\xbd\xd1\xd8C'
                     b'\x0b\xbc\x05y\x9d\xb0\x1e\xc1\xb2y:\xdfI\x983\xca$\xac\x95\xc2'
                     b'\xca\xb2L\x98\xaf\xa9cV\x04o\x82\xa7\xabj\x90\x96\x19\xab\xa1E'
                     b'M\xb62\xa6\x91\xbab\xe0\xd9\xad\xc4\x15\xda\x84i\xb8E\x1a\x81\x95'
                     b'\xca\x80\x8f>\xa9\x1b\x1a$\xb8\x9a\xe5P4\x955A\x8b\x84}\xa7'
                     b'he\xfb\x84\x15F)\xb09\xf5C\x97\x81s19\x07]\xb9h*I@\xe8\x18\xb4\xcaM'
                     b'\xcfh.\xba\xb42J0\xe7\ta\xa4\xa0\xad{`\xaa>6m\xb4\xb7\x84\xdf'
                     b'\xd6\xbd9B\xf0\xb2\x0c\x8a\t\xf4 \x15\n\x86=\xd5\x99%lvaJ\xbf\x02'
                     b'\x8bqi\x1f\r\x8d\xf1\xe7\x87\xf18\xd8b8\xbc\xf0\x90+dG\x84Lp\xb4'
                     b']vy\xca\x0e\xb3\xe3\xec\xe8\xf8\xd5<F\xbeC\x8d\x16\xe2\xde\x99'
                     b'\x97\xed\x90\xb2\xf7b\x91f\xe9"\xddM\xd3\xc5\xfc\xe5<\xdd\x8bq'
                     b'\x87\xa6\x8d\xf8E\xff\xfdr\xd6Y\xd3v~I\xd6\xf2\x7f\xae\xfc\x17'
                     b'\xb8\xb2\xa4\x1d.g\xcec\xe7\xe2\xd6\xe6\x8bh\xd6\x18\xd7\x15\xedl'
                     b'\x9e\xed\xc6\x93\x95\x14\xbe\x8e\x07\x8b\x9d\xf9\x90P\x80\xc2'
                     b'h\xbfz\x91F;h\xbaU\xf0\xf5q:\x1e\x17e\xc5-\xfez\xea\x10\xc5P'
                     b'<\xcdv\xe7\x8b\x8c(\x15O5w\xd0\x12%\x86>\xb2\xa1\r)\x90\x0b\xccC'
                     b'\xc5\x89\xc1VA\x1f]%(\x87C\x82\x91\x0e\xe9\xca\x1aE\x18\xeb/g\rX'
                     b'\x1a}\x1cJa\x05E\xcfow8\xdey\xd4b+\xd9b\x89\x16u\x81\\\xea'
                     b'\xd2\xd8v`z\x0c\xb4Px\x14\xbc\r\xca\xcbn,{u\xbd\x9dA;@]'
                     b'\xf9\xfa\x89\xa0q\x04;5\xc3E\xd7v\x1d\x9f;\x0e\x94Gi\xa0'
                     b'\xc6\xd6\x86\x15\x19\xa5\xd1\xaf\x8b\rSO\xa8=z[#PE\x97\x0eJE\x9f'
                     b'\xe8\x89\x83\xb2\xe0\xbe&\\kb\x04Q`k\xb0\xa7\x02x\x87\xf4'
                     b'\xea\xb5\x97?\xd7\xb0\xbf\xbf\xff\xdb\xd8V\xc6\xa3\xf5\xce'
                     b'\xb2\x9fKk\xa3\xe5m\xc0\xd1\xe2\x94\xbcu\xb1kd\xc7\xe3\xc2!'
                     b'\xa7m\xf1\xf8Ha\xb3\xf3\xb5?GeV\x8f\xfe\xa9\xbe2\x16x\x883\xaf\x06'
                     b'\xf6\xb9\xcd\xdc\xc1Y(*\xf0\x84\x93\x10\x92y\xd4*\xe4HD\xb0'
                     b'\x8f`s\xa2\xcef\xd7\xdd\xb0C\x9e[\xb3\xd2\x12\xb6\xdb\x8f\x9d\r'
                     b'\\\xe2T\xae\x92\xb9T\xd2\xf7\xe3\x1eQG\xa9\xe4D\x13\xde\x19G+\x0b$y'
                     b'v\xa2\xab\x9d\xd8\x86w\x9d\x92\rEI\x1dI\x1b\xdf\xd8V\xfdV'
                     b'j\x02\xf6\xc7\x04Ld\xdd\x96{z@\xb4(~K\xd3lv\xbd2\xba\xe9\x87S^\x18k'
                     b'Q\r\\\xdd\x8c\xa0\xc7\xe1\xa6\xa3\xe1\xe1\x84bl\x9cp\xb6Q+'
                     b'\xees\x10$\xa6h\xad!\xa5*G\xc1\x1a\xb8\x1e\x15\xcf9Z{\xc2V\xc6\x92'
                     b'f\xdc\x04\x88\x83\x93PuX\r\xaaY\xd2\x83pQ\x9b\xc4\xa3sE`\xd3\x9b'
                     b'\xb1M\xc2\x82\xa6\x89\xa5#\xe0\x12&\xa4#D`\xacW\xd4\xf4/'
                     b'\xa0V\x0b\x06\xb4\x9d\xf1WB2++\r>X\x9c\x1a\xa0\x9cJ\xc6\xfa\xa3@'
                     b'S\xf4\xd4ep8\xc8-\xa5\x14Q\n\xaf w\xc3\xe3\xbc\x8e\xf2\xab$Im'
                     b'\xaeH]\xe9;H%\xd1K\x0b\xe2\xf8x\x8f\x959\xd5\xecBN\x05\xa3v\xc7'
                     b'\x9c\x11\x8b8\x07P\x0b\xa6\xedG\xa3\x06-\xa2\xe8R;k\x88&P\x18\x8dV'
                     b'\x11\xb6[\xbdN\x00\xae\x1b/\xac\xe9\xba8\xfd/\x00\xd2\x85\x8f\x86'
                     b'\x8e\x12\xa3~\x8f\xee\x06,\x9b\xd8N\x18\x8c\x0c\xb3x'
                     b'\x13\x88\xd7\xdc\xf7\xdd\xa4vg\xc3\xdfv\xfaS\xe3\xf9'
                     b'\xe8\x9f\x14\x9d\n\x12\x97kp\xf5\x18{ \xdaR\xb5\xe5\x81\xb9;'
                     b'>\xed>\xdc}\x9a\xbf=\xdb\xfb*\xcf?\xfe\x95.\xc2Mq~\x84\x9f\xd2\xf7'
                     b'\xc5\xd7\xfc\xfd9~\xb6g\x8b/\xdfn^\x9e6\xdf\x1a\xf8;\xdd'
                     b'?\xe8\xdc\xe5?;_\xca\xfe\xddI\xe6/.\xb3\x93\xf3\xfe\x10\xc5'
                     b'\xc9\xdb\xb0s\xf8\xf9\xcd\x9b\xe5\xeca\xf6\xf0/\x1dt\x82&(\n'
                     b'\x00\x00')


@pytest.fixture()
def random_20_bytes():
    return b'\xe6\x97\xa5\xd0\xbc\xe6\x9c\xac\xe3\x81\xae\x20\x31\x23\x24\x25\x5e\x26\x2a'


@pytest.fixture()
def random_100_bytes():
    return (
        b'\xec\x95\x88\xe3\x81\xae\x20\x31\x23\x24\x25\x5e\x26\x2a\x7b\x7d\x5b\x5d\x3c\x3e\x3f'
        b'\x7c\x5c\x60\x7e\x21\x40\x23\x24\x25\x5e\x26\x2a\x28\x29\x5f\x2b\x2d\x3d\x7b\x7d\x5b\x5d'
        b'\x3c\x3e\x3f\x7c\x5c\x60\x7e\x21\x40\x23\x24\x25\x5e\x26\x2a\x28\x29\x5f\x2b\x2d\x3d\x7b'
        b'\x7d\x5b\x5d\x3c\x3e\x3f\x7c\x5c\x60\x7e\x21\x40'
    )


@pytest.fixture()
def random_1000_bytes():
    return (
        b'\xec\x95\x88\xeb\x85\x95\xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82\x20\x31\x23\x24'
        b'\x25\x5e\x26\x2a\x7b\x7d\x5b\x5d\x3c\x3e\x3f\x7c\x5c\x60\x7e\x21\x40\x23\x24\x25\x5e\x26'
        b'\x2a\x28\x29\x5f\x2b\x2d\x3d\x7b\x7d\x5b\x5d\x3c\x3e\x3f\x7c\x5c\x60\x7e\x21\x40\x23\x24'
        b'\x25\x5e\x26\x2a\x28\x29\x5f\x2b\x2d\x3d\x7b\x7d\x5b\x5d\x3c\x3e\x3f\x7c\x5c\x60\x7e\x21'
        b'\x40\x23\x24\x25\x5e\x26\x2a\x28\x29\x5f\x2b\x2d\x3d\x7b\x7d\x5b\x5d\x3c\x3e\x3f\x7c\x5c'
        b'\x60\x7e\x21\x40\x23\x24\x25\x5e\x26\x2a\x28\x29\x5f\x2b\x2d\x3d\x7b\x7d\x5b\x5d\x3c\x3e'
        b'\x3f\x7c\x5c\x60\x7e\x21\x40\x23\x24\x25\x5e\x26\x2a\x28\x29\x5f\x2b\x2d\x3d\x7b\x7d\x5b'
        b'\x5d\x3c\x3e\x3f\x7c\x5c\x60\x7e\x21\x40\x23\x24\x25\x5e\x26\x2a\x28\x29\x5f\x2b\x2d\x3d'
        b'\x7b\x7d\x5b\x5d\x3c\x3e\x3f\x7c\x5c\x60\x7e\x21\x40\x23\x24\x25\x5e\x26\x2a\x28\x29\x5f'
        b'\x2b\x2d\x3d\x7b\x7d\x5b\x5d\x3c\x3e\x3f\x7c\x5c\x60\x7e\x21\x40\x23\x24\x25\x5e\x26\x2a'
        b'\x28\x29\x5f\x2b\x2d\x3d\x7b\x7d\x5b\x5d\x3c\x3e\x3f\x7c\x5c\x60\x7e\x21\x40\x23\x24\x25'
        b'\x5e\x26\x2a\x28\x29\x5f\x2b\x2d\x3d\x7b\x7d\x5b\x5d\x3c\x3e\x3f\x7c\x5c\x60\x7e\x21\x40'
        b'\x23\x24\x25\x5e\x26\x2a\x28\x29\x5f\x2b\x2d\x3d\x7b\x7d\x5b\x5d\x3c\x3e\x3f\x7c\x5c\x60'
        b'\x7e\x21\x40\x23\x24\x25\x5e\x26\x2a\x28\x29\x5f\x2b\x2d\x3d\x7b\x7d\x5b\x5d\x3c\x3e\x3f'
        b'\x7c\x5c\x60\x7e\x21\x40\x23\x24\x25\x5e\x26\x2a\x28\x29\x5f\x2b\x2d\x3d\x7b\x7d\x5b\x5d'
        b'\x3c\x3e\x3f\x7c\x5c\x60\x7e\x21\x40'
    )


@pytest.mark.unittest
class TestMetadataLSB:
    def test_read_lsb_metadata(self, nai3_file, metadata_expected):
        assert read_lsb_metadata(nai3_file) == metadata_expected

    def test_read_lsb_metadata_empty(self, nai3_info_rgb_file, nai3_clear_rgba_file, nai3_clear_file):
        with pytest.raises(LSBReadError):
            _ = read_lsb_metadata(nai3_info_rgb_file)
        with pytest.raises(LSBReadError):
            _ = read_lsb_metadata(nai3_clear_file)
        with pytest.raises(LSBReadError):
            _ = read_lsb_metadata(nai3_clear_rgba_file)

    def test_read_lsb_raw_bytes(self, nai3_file, raw_bytes_expected):
        assert read_lsb_raw_bytes(nai3_file) == raw_bytes_expected

    def test_read_lsb_raw_bytes_empty(self, nai3_info_rgb_file, nai3_clear_rgba_file, nai3_clear_file):
        with pytest.raises(LSBReadError):
            _ = read_lsb_raw_bytes(nai3_info_rgb_file)
        with pytest.raises(LSBReadError):
            _ = read_lsb_raw_bytes(nai3_clear_file)
        with pytest.raises(LSBReadError):
            _ = read_lsb_raw_bytes(nai3_clear_rgba_file)

    def test_write_lsb_raw_bytes_random_20(self, nai3_clear_file, random_20_bytes):
        with pytest.raises(LSBReadError):
            _ = read_lsb_raw_bytes(nai3_clear_file)

        image = write_lsb_raw_bytes(nai3_clear_file, random_20_bytes)
        assert read_lsb_raw_bytes(image) == random_20_bytes

        with isolated_directory():
            image.save('image.png')
            assert read_lsb_raw_bytes('image.png') == random_20_bytes

    def test_write_lsb_raw_bytes_random_100(self, nai3_clear_file, random_100_bytes):
        with pytest.raises(LSBReadError):
            _ = read_lsb_raw_bytes(nai3_clear_file)

        image = write_lsb_raw_bytes(nai3_clear_file, random_100_bytes)
        assert read_lsb_raw_bytes(image) == random_100_bytes

        with isolated_directory():
            image.save('image.png')
            assert read_lsb_raw_bytes('image.png') == random_100_bytes

    def test_write_lsb_raw_bytes_random_1000(self, nai3_clear_file, random_1000_bytes):
        with pytest.raises(LSBReadError):
            _ = read_lsb_raw_bytes(nai3_clear_file)

        image = write_lsb_raw_bytes(nai3_clear_file, random_1000_bytes)
        assert read_lsb_raw_bytes(image) == random_1000_bytes

        with isolated_directory():
            image.save('image.png')
            assert read_lsb_raw_bytes('image.png') == random_1000_bytes

    def test_write_lsb_metadata_json(self, nai3_clear_file, metadata_expected):
        with pytest.raises(LSBReadError):
            _ = read_lsb_metadata(nai3_clear_file)

        image = write_lsb_metadata(nai3_clear_file, metadata_expected)
        assert read_lsb_metadata(image) == metadata_expected

        with isolated_directory():
            image.save('image.png')
            assert read_lsb_metadata('image.png') == metadata_expected

    def test_write_lsb_metadata_pnginfo(self, nai3_clear_file, pnginfo_expected, metadata_expected):
        with pytest.raises(LSBReadError):
            _ = read_lsb_metadata(nai3_clear_file)

        image = write_lsb_metadata(nai3_clear_file, pnginfo_expected)
        assert read_lsb_metadata(image) == metadata_expected

        with isolated_directory():
            image.save('image.png')
            assert read_lsb_metadata('image.png') == metadata_expected

    def test_write_lsb_metadata_bytes(self, nai3_clear_file, random_1000_bytes):
        with pytest.raises(LSBReadError):
            _ = read_lsb_metadata(nai3_clear_file)

        image = write_lsb_metadata(nai3_clear_file, random_1000_bytes)
        assert read_lsb_raw_bytes(image) == random_1000_bytes

        with isolated_directory():
            image.save('image.png')
            assert read_lsb_raw_bytes('image.png') == random_1000_bytes
