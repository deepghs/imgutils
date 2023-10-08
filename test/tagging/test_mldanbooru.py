import pytest

from imgutils.tagging import get_mldanbooru_tags
from imgutils.tagging.mldanbooru import _open_mldanbooru_model
from test.testings import get_testfile


@pytest.fixture()
def _release_model_after_run():
    try:
        yield
    finally:
        _open_mldanbooru_model.cache_clear()


@pytest.mark.unittest
class TestTaggingMldanbooru:
    @pytest.mark.parametrize(['keep_ratio'], [(True,), (False,)])
    def test_get_mldanbooru_tags(self, keep_ratio):
        tags = get_mldanbooru_tags(get_testfile('6124220.jpg'), keep_ratio=keep_ratio)
        assert tags['cat'] >= 0.8

        tags = get_mldanbooru_tags(get_testfile('6125785.jpg'), keep_ratio=keep_ratio)
        assert tags['1girl'] >= 0.95

    def test_get_mldanbooru_tags_sample(self):
        tags = get_mldanbooru_tags(get_testfile('nude_girl.png'))
        assert tags == pytest.approx({
            '1girl': 0.9999977350234985, 'breasts': 0.999940037727356, 'nipples': 0.999920129776001,
            'solo': 0.9993574023246765, 'pussy': 0.9993218183517456, 'horns': 0.9977452158927917,
            'nude': 0.995971143245697, 'purple_eyes': 0.9957809448242188, 'long_hair': 0.9929291605949402,
            'navel': 0.9814828038215637, 'armpits': 0.9808009266853333, 'spread_legs': 0.9767358303070068,
            'pussy_juice': 0.959962785243988, 'blush': 0.9482676386833191, 'uncensored': 0.9446588158607483,
            'looking_at_viewer': 0.9295657873153687, 'red_hair': 0.919776201248169,
            'medium_breasts': 0.9020175337791443, 'completely_nude': 0.8965569138526917, 'arms_up': 0.8882529139518738,
            'on_back': 0.8701885342597961, 'arms_behind_head': 0.8692260980606079, 'lying': 0.8653205037117004,
            'pillow': 0.8645844459533691, 'bangs': 0.8618668913841248, 'smile': 0.8531544804573059,
            'very_long_hair': 0.8332053422927856, 'pointy_ears': 0.8194612264633179, 'stomach': 0.8194073438644409,
            'hair_intakes': 0.8191318511962891, 'on_bed': 0.8055890202522278, 'sweat': 0.7933878302574158,
            'thighs': 0.7835342884063721, 'hair_between_eyes': 0.7693091630935669,
            'eyebrows_visible_through_hair': 0.7672545313835144, 'closed_mouth': 0.7638942003250122,
            'breasts_apart': 0.7527053952217102, 'bed': 0.7515304088592529, 'slit_pupils': 0.7464283108711243,
            'barefoot': 0.7429600954055786, 'bed_sheet': 0.7186222076416016, 'fang': 0.7162102460861206,
            'clitoris': 0.7013473510742188,
        }, abs=1e-3)

    def test_get_mldanbooru_tags_no_overlap(self):
        tags = get_mldanbooru_tags(get_testfile('nude_girl.png'), drop_overlap=True)
        assert tags == pytest.approx({
            '1girl': 0.9999977350234985, 'nipples': 0.999920129776001, 'solo': 0.9993574023246765,
            'horns': 0.9977452158927917, 'purple_eyes': 0.9957809448242188, 'navel': 0.9814828038215637,
            'armpits': 0.9808009266853333, 'spread_legs': 0.9767358303070068, 'pussy_juice': 0.959962785243988,
            'blush': 0.9482676386833191, 'uncensored': 0.9446586966514587, 'looking_at_viewer': 0.9295657873153687,
            'red_hair': 0.9197760820388794, 'medium_breasts': 0.9020175337791443, 'completely_nude': 0.8965569138526917,
            'arms_up': 0.8882529139518738, 'on_back': 0.8701885342597961, 'arms_behind_head': 0.8692260980606079,
            'pillow': 0.8645844459533691, 'bangs': 0.8618668913841248, 'smile': 0.8531544804573059,
            'very_long_hair': 0.8332052230834961, 'pointy_ears': 0.8194612264633179, 'stomach': 0.8194073438644409,
            'hair_intakes': 0.8191318511962891, 'on_bed': 0.8055890202522278, 'sweat': 0.793387770652771,
            'thighs': 0.7835341691970825, 'hair_between_eyes': 0.7693091034889221,
            'eyebrows_visible_through_hair': 0.7672545909881592, 'closed_mouth': 0.7638942003250122,
            'breasts_apart': 0.7527053356170654, 'slit_pupils': 0.7464284300804138, 'barefoot': 0.7429600358009338,
            'bed_sheet': 0.7186222672462463, 'fang': 0.7162103652954102, 'clitoris': 0.7013473510742188
        }, abs=1e-3)
