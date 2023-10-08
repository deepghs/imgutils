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
        assert tags['cat'] >= 0.8
        assert not chars

        rating, tags, chars = get_wd14_tags(get_testfile('6125785.jpg'))
        assert 0.55 <= rating['general'] <= 0.65
        assert 0.35 <= rating['sensitive'] <= 0.45
        assert tags['1girl'] >= 0.95
        assert chars['hu_tao_(genshin_impact)'] >= 0.95

    def test_wd14_tags_sample(self):
        rating, tags, chars = get_wd14_tags(get_testfile('nude_girl.png'))
        assert rating == pytest.approx({
            'general': 0.0020540356636047363,
            'sensitive': 0.0080718994140625,
            'questionable': 0.003170192241668701,
            'explicit': 0.984081506729126,
        }, abs=1e-3)
        assert tags == pytest.approx({
            '1girl': 0.998362123966217, 'solo': 0.9912548065185547, 'long_hair': 0.9401906728744507,
            'breasts': 0.983635425567627, 'looking_at_viewer': 0.9146994352340698, 'blush': 0.8892400860786438,
            'smile': 0.43393653631210327, 'bangs': 0.49712443351745605, 'large_breasts': 0.5196534395217896,
            'navel': 0.9653235077857971, 'hair_between_eyes': 0.5786703824996948, 'very_long_hair': 0.8142435550689697,
            'closed_mouth': 0.9369247555732727, 'nipples': 0.9660118222236633, 'purple_eyes': 0.9676010012626648,
            'collarbone': 0.588348925113678, 'nude': 0.9496222734451294, 'red_hair': 0.9200156331062317,
            'sweat': 0.8690457344055176, 'horns': 0.9711267948150635, 'pussy': 0.9868264198303223,
            'spread_legs': 0.9603149890899658, 'armpits': 0.9024748802185059, 'stomach': 0.6723923087120056,
            'arms_up': 0.9380699396133423, 'completely_nude': 0.9002960920333862, 'uncensored': 0.8612104058265686,
            'pussy_juice': 0.6021570563316345, 'feet_out_of_frame': 0.39779460430145264, 'on_bed': 0.610720157623291,
            'arms_behind_head': 0.44814401865005493, 'breasts_apart': 0.39798974990844727,
            'clitoris': 0.5310801267623901
        }, abs=1e-3)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9942929744720459}, abs=1e-3)

    def test_wd14_tags_no_overlap(self):
        rating, tags, chars = get_wd14_tags(get_testfile('nude_girl.png'), drop_overlap=True)
        # print(tags)
        assert rating == pytest.approx({
            'general': 0.0020540356636047363,
            'sensitive': 0.0080718994140625,
            'questionable': 0.003170192241668701,
            'explicit': 0.984081506729126,
        }, abs=1e-3)
        assert tags == pytest.approx({
            '1girl': 0.998362123966217, 'solo': 0.9912548065185547, 'looking_at_viewer': 0.9146994352340698,
            'blush': 0.8892400860786438, 'smile': 0.43393653631210327, 'bangs': 0.49712443351745605,
            'large_breasts': 0.5196534395217896, 'navel': 0.9653235077857971, 'hair_between_eyes': 0.5786703824996948,
            'very_long_hair': 0.8142435550689697, 'closed_mouth': 0.9369247555732727, 'nipples': 0.9660118222236633,
            'purple_eyes': 0.9676010012626648, 'collarbone': 0.588348925113678, 'red_hair': 0.9200156331062317,
            'sweat': 0.8690457344055176, 'horns': 0.9711267948150635, 'spread_legs': 0.9603149890899658,
            'armpits': 0.9024748802185059, 'stomach': 0.6723923087120056, 'arms_up': 0.9380699396133423,
            'completely_nude': 0.9002960920333862, 'uncensored': 0.8612104058265686, 'pussy_juice': 0.6021570563316345,
            'feet_out_of_frame': 0.39779460430145264, 'on_bed': 0.610720157623291,
            'arms_behind_head': 0.44814401865005493, 'breasts_apart': 0.39798974990844727,
            'clitoris': 0.5310801267623901
        }, abs=1e-3)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9942929744720459}, abs=1e-3)
