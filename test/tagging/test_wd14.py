import pytest

from imgutils.tagging import get_wd14_tags
from imgutils.tagging.wd14 import _get_wd14_model
from test.testings import get_testfile


@pytest.fixture(autouse=True, scope='module')
def _release_model_after_run():
    try:
        yield
    finally:
        _get_wd14_model.cache_clear()


@pytest.mark.unittest
class TestTaggingWd14:
    def test_get_wd14_tags(self):
        rating, tags, chars = get_wd14_tags(get_testfile('6124220.jpg'))

        assert rating['general'] > 0.9
        assert tags['cat_girl'] >= 0.8
        assert not chars
        assert isinstance(rating['general'], float)
        assert isinstance(tags['cat_girl'], float)

        rating, tags, chars = get_wd14_tags(get_testfile('6125785.jpg'))
        assert 0.6 <= rating['general'] <= 0.8
        assert tags['1girl'] >= 0.95
        assert chars['hu_tao_(genshin_impact)'] >= 0.95
        assert isinstance(rating['general'], float)
        assert isinstance(tags['1girl'], float)
        assert isinstance(chars['hu_tao_(genshin_impact)'], float)

    def test_wd14_tags_sample(self):
        rating, tags, chars = get_wd14_tags(get_testfile('nude_girl.png'))

        assert rating == pytest.approx({
            'general': 0.00043779611587524414,
            'sensitive': 0.002305924892425537,
            'questionable': 0.0011759400367736816,
            'explicit': 0.9944100975990295
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.9993372559547424, 'solo': 0.9911975264549255, 'long_hair': 0.9677416086196899,
            'breasts': 0.9896668195724487, 'looking_at_viewer': 0.9188930988311768, 'blush': 0.9104846119880676,
            'smile': 0.5376636981964111, 'navel': 0.9513114094734192, 'hair_between_eyes': 0.4308745265007019,
            'closed_mouth': 0.759302020072937, 'very_long_hair': 0.6630508303642273,
            'medium_breasts': 0.6663066148757935, 'nipples': 0.9911118149757385, 'purple_eyes': 0.9750030040740967,
            'thighs': 0.529353678226471, 'sweat': 0.6274301409721375, 'red_hair': 0.9703063368797302,
            'nude': 0.9724845290184021, 'lying': 0.690057635307312, 'horns': 0.9886922836303711,
            'pussy': 0.9820598363876343, 'spread_legs': 0.9256478548049927, 'stomach': 0.8168477416038513,
            'on_back': 0.5197966694831848, 'armpits': 0.9639391303062439, 'arms_up': 0.9117614030838013,
            'completely_nude': 0.8872356414794922, 'pillow': 0.7360897660255432, 'uncensored': 0.9299367666244507,
            'pussy_juice': 0.8235344886779785, 'on_bed': 0.7741400003433228, 'hair_intakes': 0.4976382851600647,
            'demon_horns': 0.5313447117805481, 'arms_behind_head': 0.5415608882904053,
            'breasts_apart': 0.35740798711776733, 'clitoris': 0.44502270221710205
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9957615733146667}, abs=2e-2)

    def test_wd14_tags_sample_no_underline(self):
        rating, tags, chars = get_wd14_tags(get_testfile('nude_girl.png'), no_underline=True)
        assert rating == pytest.approx({
            'general': 0.00043779611587524414,
            'sensitive': 0.002305924892425537,
            'questionable': 0.0011759400367736816,
            'explicit': 0.9944100975990295
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.9993372559547424, 'solo': 0.9911975264549255, 'long hair': 0.9677416086196899,
            'breasts': 0.9896668195724487, 'looking at viewer': 0.9188930988311768, 'blush': 0.9104846119880676,
            'smile': 0.5376636981964111, 'navel': 0.9513114094734192, 'hair between eyes': 0.4308745265007019,
            'closed mouth': 0.759302020072937, 'very long hair': 0.6630508303642273,
            'medium breasts': 0.6663066148757935, 'nipples': 0.9911118149757385, 'purple eyes': 0.9750030040740967,
            'thighs': 0.529353678226471, 'sweat': 0.6274301409721375, 'red hair': 0.9703063368797302,
            'nude': 0.9724845290184021, 'lying': 0.690057635307312, 'horns': 0.9886922836303711,
            'pussy': 0.9820598363876343, 'spread legs': 0.9256478548049927, 'stomach': 0.8168477416038513,
            'on back': 0.5197966694831848, 'armpits': 0.9639391303062439, 'arms up': 0.9117614030838013,
            'completely nude': 0.8872356414794922, 'pillow': 0.7360897660255432, 'uncensored': 0.9299367666244507,
            'pussy juice': 0.8235344886779785, 'on bed': 0.7741400003433228, 'hair intakes': 0.4976382851600647,
            'demon horns': 0.5313447117805481, 'arms behind head': 0.5415608882904053,
            'breasts apart': 0.35740798711776733, 'clitoris': 0.44502270221710205
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr (arknights)': 0.9957615733146667}, abs=2e-2)

    def test_wd14_tags_sample_mcut(self):
        rating, tags, chars = get_wd14_tags(
            get_testfile('nude_girl.png'),
            general_mcut_enabled=True,
            character_mcut_enabled=True,
        )
        assert rating == pytest.approx({
            'general': 0.00043779611587524414,
            'sensitive': 0.002305924892425537,
            'questionable': 0.0011759400367736816,
            'explicit': 0.9944100975990295
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.9993372559547424, 'solo': 0.9911975264549255, 'long_hair': 0.9677416086196899,
            'breasts': 0.9896668195724487, 'looking_at_viewer': 0.9188930988311768, 'blush': 0.9104846119880676,
            'navel': 0.9513114094734192, 'closed_mouth': 0.759302020072937, 'very_long_hair': 0.6630508303642273,
            'medium_breasts': 0.6663066148757935, 'nipples': 0.9911118149757385, 'purple_eyes': 0.9750030040740967,
            'sweat': 0.6274301409721375, 'red_hair': 0.9703063368797302, 'nude': 0.9724845290184021,
            'lying': 0.690057635307312, 'horns': 0.9886922836303711, 'pussy': 0.9820598363876343,
            'spread_legs': 0.9256478548049927, 'stomach': 0.8168477416038513, 'armpits': 0.9639391303062439,
            'arms_up': 0.9117614030838013, 'completely_nude': 0.8872356414794922, 'pillow': 0.7360897660255432,
            'uncensored': 0.9299367666244507, 'pussy_juice': 0.8235344886779785, 'on_bed': 0.7741400003433228
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9957615733146667}, abs=2e-2)

    def test_wd14_tags_no_overlap(self):
        rating, tags, chars = get_wd14_tags(get_testfile('nude_girl.png'), drop_overlap=True)
        assert rating == pytest.approx({
            'general': 0.00043779611587524414,
            'sensitive': 0.002305924892425537,
            'questionable': 0.0011759400367736816,
            'explicit': 0.9944100975990295
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.9993372559547424, 'solo': 0.9911975264549255, 'looking_at_viewer': 0.9188930988311768,
            'blush': 0.9104846119880676, 'smile': 0.5376636981964111, 'navel': 0.9513114094734192,
            'hair_between_eyes': 0.4308745265007019, 'closed_mouth': 0.759302020072937,
            'very_long_hair': 0.6630508303642273, 'medium_breasts': 0.6663066148757935, 'nipples': 0.9911118149757385,
            'purple_eyes': 0.9750030040740967, 'thighs': 0.5293537378311157, 'sweat': 0.6274301409721375,
            'red_hair': 0.9703063368797302, 'spread_legs': 0.9256478548049927, 'stomach': 0.8168477416038513,
            'on_back': 0.5197967290878296, 'armpits': 0.9639391303062439, 'arms_up': 0.9117614030838013,
            'completely_nude': 0.8872356414794922, 'pillow': 0.7360897660255432, 'uncensored': 0.9299367666244507,
            'pussy_juice': 0.8235344886779785, 'on_bed': 0.7741400003433228, 'hair_intakes': 0.4976382851600647,
            'demon_horns': 0.5313447117805481, 'arms_behind_head': 0.5415608882904053,
            'breasts_apart': 0.35740798711776733, 'clitoris': 0.44502270221710205
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9957615733146667}, abs=2e-2)

    def test_wd14_rgba(self):
        rating, tags, chars = get_wd14_tags(get_testfile('nian.png'))
        assert rating == pytest.approx({
            'general': 0.013875722885131836, 'sensitive': 0.9790834188461304,
            'questionable': 0.0004328787326812744, 'explicit': 0.00010639429092407227,
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.996912956237793, 'solo': 0.9690700769424438, 'long_hair': 0.9183608293533325,
            'breasts': 0.5793432593345642, 'looking_at_viewer': 0.9029998779296875, 'smile': 0.7181373834609985,
            'open_mouth': 0.5431916117668152, 'simple_background': 0.3519788384437561,
            'long_sleeves': 0.7442969679832458, 'white_background': 0.6004813313484192, 'holding': 0.7325218319892883,
            'navel': 0.9297535419464111, 'jewelry': 0.5435991287231445, 'standing': 0.8762419819831848,
            'purple_eyes': 0.9269286394119263, 'tail': 0.8547350168228149, 'full_body': 0.9316157102584839,
            'white_hair': 0.9207442402839661, 'braid': 0.37353646755218506, 'multicolored_hair': 0.6516135931015015,
            'thighs': 0.451822429895401, ':d': 0.5130974054336548, 'red_hair': 0.5783762335777283,
            'small_breasts': 0.3563075065612793, 'boots': 0.6243380308151245, 'open_clothes': 0.8822896480560303,
            'horns': 0.965097188949585, 'shorts': 0.9586330056190491, 'shoes': 0.4847032427787781,
            'socks': 0.47281092405319214, 'tongue': 0.9029147624969482, 'pointy_ears': 0.8633939623832703,
            'belt': 0.4783763289451599, 'midriff': 0.9044876098632812, 'tongue_out': 0.9018264412879944,
            'wide_sleeves': 0.7076666951179504, 'stomach': 0.891795814037323, 'streaked_hair': 0.6510426998138428,
            'coat': 0.7965987324714661, 'crop_top': 0.6840215921401978, 'hand_on_own_hip': 0.5604047179222107,
            'strapless': 0.950110137462616, 'short_shorts': 0.6481347680091858, 'bare_legs': 0.5356456637382507,
            'white_footwear': 0.8399633169174194, 'transparent_background': 0.3643641471862793, ':p': 0.532076358795166,
            'half_updo': 0.5155724883079529, 'open_coat': 0.8147380352020264, 'beads': 0.3977043032646179,
            'white_shorts': 0.9007017612457275, 'white_coat': 0.8003122806549072, 'bandeau': 0.9671074151992798,
            'tube_top': 0.9783295392990112, 'bead_bracelet': 0.3510066270828247, 'red_bandeau': 0.8741766214370728
        }, abs=2e-2)
        assert chars == pytest.approx({'nian_(arknights)': 0.9968841671943665}, abs=2e-2)
