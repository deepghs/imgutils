import numpy as np
import pytest

from imgutils.tagging import get_pixai_tags
from imgutils.tagging.pixai import _open_tags, _open_preprocess, _open_onnx_model, _open_default_category_thresholds
from test.testings import get_testfile


@pytest.fixture(autouse=True, scope='module')
def _release_model_after_run():
    try:
        yield
    finally:
        _open_tags.cache_clear()
        _open_preprocess.cache_clear()
        _open_onnx_model.cache_clear()
        _open_default_category_thresholds.cache_clear()


@pytest.mark.unittest
class TestTaggingPixAI:
    def test_get_pixai_tags(self):
        tags, chars = get_pixai_tags(get_testfile('6124220.jpg'))

        assert tags['cat_girl'] >= 0.8
        assert not chars
        assert isinstance(tags['cat_girl'], float)

        tags, chars = get_pixai_tags(get_testfile('6125785.jpg'))
        assert tags['1girl'] >= 0.90
        assert chars['hu_tao_(genshin_impact)'] >= 0.95
        assert isinstance(tags['1girl'], float)
        assert isinstance(chars['hu_tao_(genshin_impact)'], float)

    def test_pixai_tags_sample(self):
        tags, chars = get_pixai_tags(get_testfile('6125785.jpg'))

        assert tags == pytest.approx({
            'symbol-shaped_pupils': 0.9938011169433594, 'ghost': 0.9909012317657471,
            'flower-shaped_pupils': 0.9810856580734253, 'porkpie_hat': 0.9693692922592163,
            'black_nails': 0.9692082405090332, 'ghost_pose': 0.9579154253005981, 'ring': 0.9509532451629639,
            '1girl': 0.9435760974884033, 'hat_flower': 0.9315004348754883, 'hat': 0.9308193922042847,
            'flower': 0.9203469753265381, 'plum_blossoms': 0.8814681768417358, 'red_shirt': 0.8734415769577026,
            'jewelry': 0.8436187505722046, 'claw_pose': 0.8382970094680786, 'chinese_clothes': 0.8213527202606201,
            'long_sleeves': 0.8101956844329834, 'long_hair': 0.7956619262695312, 'twintails': 0.7865841388702393,
            'looking_at_viewer': 0.7795377373695374, 'hat_tassel': 0.7775378227233887,
            'multiple_rings': 0.7483937740325928, 'signature': 0.7478364109992981, 'smile': 0.741124153137207,
            'open_mouth': 0.7302751541137695, 'red_flower': 0.705192506313324, 'solo': 0.6761863827705383,
            'hat_ornament': 0.6489882469177246, 'red_eyes': 0.6290297508239746, 'black_hat': 0.6113986372947693,
            'brown_coat': 0.6067467927932739, 'nail_polish': 0.5937596559524536, 'coat': 0.5664999485015869,
            'upper_body': 0.5661808848381042, 'brown_hair': 0.5366549491882324, 'thumb_ring': 0.5287110209465027,
            'star_(symbol)': 0.5028221011161804, 'shirt': 0.4754156470298767, 'star-shaped_pupils': 0.45203927159309387,
            'hair_between_eyes': 0.4512987434864044, 'orange_eyes': 0.4298587739467621, ':d': 0.4219313859939575,
            'tangzhuang': 0.3894977569580078, ':3': 0.3866003453731537, 'blush': 0.3761531412601471,
            'v-shaped_eyebrows': 0.34583818912506104, 'sidelocks': 0.3450319468975067,
            'brown_shirt': 0.31902527809143066, 'blurry': 0.3167526423931122
        }, abs=2e-2)
        assert chars == pytest.approx({
            'hu_tao_(genshin_impact)': 0.9999773502349854, 'boo_tao_(genshin_impact)': 0.9996482133865356
        }, abs=2e-2)

    def test_pixai_rgba(self):
        tags, chars = get_pixai_tags(get_testfile('nian.png'))
        assert tags == pytest.approx({
            'red_tube_top': 0.9998065233230591, 'tube_top': 0.9994766712188721, 'red_bandeau': 0.9992805123329163,
            'white_shorts': 0.9976104497909546, 'bandeau': 0.9968563914299011,
            'transparent_background': 0.9929141402244568, 'horns': 0.9903750419616699, 'full_body': 0.9860043525695801,
            'midriff': 0.9834859371185303, 'shorts': 0.9828792810440063, '1girl': 0.982780933380127,
            'strapless': 0.9817174673080444, 'navel': 0.9786539673805237, 'white_footwear': 0.9764545559883118,
            'white_hair': 0.9713669419288635, 'solo': 0.9699287414550781, 'tail': 0.9654039740562439,
            'stomach': 0.9624348282814026, 'standing': 0.9548828601837158, 'pointy_ears': 0.9536501169204712,
            'open_coat': 0.9320983290672302, 'looking_at_viewer': 0.9311813116073608,
            'open_clothes': 0.9311613440513611, 'long_hair': 0.9210299253463745, 'hand_on_own_hip': 0.9110379219055176,
            'holding': 0.9101808071136475, 'short_shorts': 0.9012227058410645, 'tongue': 0.8910117745399475,
            'tongue_out': 0.8871811628341675, 'dragon_horns': 0.8710488080978394, 'weapon': 0.8667179346084595,
            'coat': 0.8642098307609558, 'belt': 0.8602883219718933, 'tachi-e': 0.8591709136962891,
            'crop_top': 0.856552004814148, 'white_coat': 0.8258588314056396, 'dragon_tail': 0.822641909122467,
            'purple_eyes': 0.7998863458633423, 'streaked_hair': 0.796852707862854, 'half_updo': 0.790149450302124,
            'thighs': 0.7805066108703613, 'bare_legs': 0.7744182348251343, ':d': 0.7707932591438293,
            'wide_sleeves': 0.767501711845398, 'smile': 0.765789270401001, 'multicolored_hair': 0.7620936632156372,
            'bead_bracelet': 0.7511173486709595, 'red_hair': 0.7497332692146301, 'drop_shadow': 0.7467425465583801,
            'long_sleeves': 0.7251672744750977, 'red_skin': 0.7151302099227905, 'sidelocks': 0.6788227558135986,
            'socks': 0.6740190982818604, ':p': 0.6698598861694336, 'sword': 0.6676492691040039,
            'jewelry': 0.6632057428359985, 'hand_up': 0.637663722038269, 'boots': 0.6372023224830627,
            'black_socks': 0.626619815826416, 'pixel_art': 0.6166024208068848, 'black_belt': 0.6042207479476929,
            'holding_weapon': 0.6014618873596191, 'breasts': 0.6006667017936707, 'ankle_boots': 0.5968820452690125,
            'white_jacket': 0.5880962014198303, 'jacket': 0.5767049789428711, 'tassel_earrings': 0.5749104022979736,
            'open_mouth': 0.5724694132804871, 'beads': 0.5716468095779419, 'braid': 0.5678756237030029,
            'open_jacket': 0.5464628338813782, 'bare_shoulders': 0.5238108038902283, 'shoes': 0.506024181842804,
            'bracelet': 0.4898264706134796, 'pouch': 0.48912930488586426, 'colored_skin': 0.46792706847190857,
            'originium_arts_(arknights)': 0.4517717957496643, 'dragon_girl': 0.4428599774837494,
            'thigh_strap': 0.4376051723957062, 'earrings': 0.4229695200920105, 'collarbone': 0.41589391231536865,
            'red_eyes': 0.4031139016151428, 'body_markings': 0.4004215598106384, 'medium_breasts': 0.384432315826416,
            'holding_sword': 0.3541863262653351, 'white_pants': 0.34769535064697266,
            'club_(weapon)': 0.33692681789398193, 'eyeshadow': 0.3357437252998352, 'ponytail': 0.3333626389503479,
            'tassel': 0.33294352889060974, 'multiple_weapons': 0.3174567222595215, 'pink_eyes': 0.31579816341400146,
            'flame-tipped_tail': 0.3054245710372925
        }, abs=2e-2)
        assert chars == pytest.approx({'nian_(arknights)': 0.9999774694442749}, abs=2e-2)

    def test_pixai_tags_sample_ips(self):
        tags, chars, ips, ips_count, ips_mapping = get_pixai_tags(
            get_testfile('6125785.jpg'),
            fmt=('general', 'character', 'ips', 'ips_count', 'ips_mapping')
        )

        assert tags == pytest.approx({
            'symbol-shaped_pupils': 0.9938011169433594, 'ghost': 0.9909012317657471,
            'flower-shaped_pupils': 0.9810856580734253, 'porkpie_hat': 0.9693692922592163,
            'black_nails': 0.9692082405090332, 'ghost_pose': 0.9579154253005981, 'ring': 0.9509532451629639,
            '1girl': 0.9435760974884033, 'hat_flower': 0.9315004348754883, 'hat': 0.9308193922042847,
            'flower': 0.9203469753265381, 'plum_blossoms': 0.8814681768417358, 'red_shirt': 0.8734415769577026,
            'jewelry': 0.8436187505722046, 'claw_pose': 0.8382970094680786, 'chinese_clothes': 0.8213527202606201,
            'long_sleeves': 0.8101956844329834, 'long_hair': 0.7956619262695312, 'twintails': 0.7865841388702393,
            'looking_at_viewer': 0.7795377373695374, 'hat_tassel': 0.7775378227233887,
            'multiple_rings': 0.7483937740325928, 'signature': 0.7478364109992981, 'smile': 0.741124153137207,
            'open_mouth': 0.7302751541137695, 'red_flower': 0.705192506313324, 'solo': 0.6761863827705383,
            'hat_ornament': 0.6489882469177246, 'red_eyes': 0.6290297508239746, 'black_hat': 0.6113986372947693,
            'brown_coat': 0.6067467927932739, 'nail_polish': 0.5937596559524536, 'coat': 0.5664999485015869,
            'upper_body': 0.5661808848381042, 'brown_hair': 0.5366549491882324, 'thumb_ring': 0.5287110209465027,
            'star_(symbol)': 0.5028221011161804, 'shirt': 0.4754156470298767, 'star-shaped_pupils': 0.45203927159309387,
            'hair_between_eyes': 0.4512987434864044, 'orange_eyes': 0.4298587739467621, ':d': 0.4219313859939575,
            'tangzhuang': 0.3894977569580078, ':3': 0.3866003453731537, 'blush': 0.3761531412601471,
            'v-shaped_eyebrows': 0.34583818912506104, 'sidelocks': 0.3450319468975067,
            'brown_shirt': 0.31902527809143066, 'blurry': 0.3167526423931122
        }, abs=2e-2)
        assert chars == pytest.approx({
            'hu_tao_(genshin_impact)': 0.9999773502349854, 'boo_tao_(genshin_impact)': 0.9996482133865356
        }, abs=2e-2)
        assert ips == ['genshin_impact']
        assert ips_count == {'genshin_impact': 2}
        assert ips_mapping == {
            'hu_tao_(genshin_impact)': ['genshin_impact'],
            'boo_tao_(genshin_impact)': ['genshin_impact'],
        }

    def test_pixai_tags_sample_custom(self):
        tags, chars, embedding, logits, prediction = get_pixai_tags(
            get_testfile('6125785.jpg'),
            fmt=('general', 'character', 'embedding', 'logits', 'prediction'),
            model_name='deepghs/pixai-tagger-v0.9-onnx',
        )

        assert tags == pytest.approx({
            'symbol-shaped_pupils': 0.9938011169433594, 'ghost': 0.9909012317657471,
            'flower-shaped_pupils': 0.9810856580734253, 'porkpie_hat': 0.9693692922592163,
            'black_nails': 0.9692082405090332, 'ghost_pose': 0.9579154253005981, 'ring': 0.9509532451629639,
            '1girl': 0.9435760974884033, 'hat_flower': 0.9315004348754883, 'hat': 0.9308193922042847,
            'flower': 0.9203469753265381, 'plum_blossoms': 0.8814681768417358, 'red_shirt': 0.8734415769577026,
            'jewelry': 0.8436187505722046, 'claw_pose': 0.8382970094680786, 'chinese_clothes': 0.8213527202606201,
            'long_sleeves': 0.8101956844329834, 'long_hair': 0.7956619262695312, 'twintails': 0.7865841388702393,
            'looking_at_viewer': 0.7795377373695374, 'hat_tassel': 0.7775378227233887,
            'multiple_rings': 0.7483937740325928, 'signature': 0.7478364109992981, 'smile': 0.741124153137207,
            'open_mouth': 0.7302751541137695, 'red_flower': 0.705192506313324, 'solo': 0.6761863827705383,
            'hat_ornament': 0.6489882469177246, 'red_eyes': 0.6290297508239746, 'black_hat': 0.6113986372947693,
            'brown_coat': 0.6067467927932739, 'nail_polish': 0.5937596559524536, 'coat': 0.5664999485015869,
            'upper_body': 0.5661808848381042, 'brown_hair': 0.5366549491882324, 'thumb_ring': 0.5287110209465027,
            'star_(symbol)': 0.5028221011161804, 'shirt': 0.4754156470298767, 'star-shaped_pupils': 0.45203927159309387,
            'hair_between_eyes': 0.4512987434864044, 'orange_eyes': 0.4298587739467621, ':d': 0.4219313859939575,
            'tangzhuang': 0.3894977569580078, ':3': 0.3866003453731537, 'blush': 0.3761531412601471,
            'v-shaped_eyebrows': 0.34583818912506104, 'sidelocks': 0.3450319468975067,
            'brown_shirt': 0.31902527809143066, 'blurry': 0.3167526423931122
        }, abs=2e-2)
        assert chars == pytest.approx({
            'hu_tao_(genshin_impact)': 0.9999773502349854, 'boo_tao_(genshin_impact)': 0.9996482133865356
        }, abs=2e-2)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
        assert isinstance(logits, np.ndarray)
        assert logits.shape == (13461,)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (13461,)
