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
