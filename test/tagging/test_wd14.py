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
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.998561441898346, 'solo': 0.9918843507766724, 'long_hair': 0.9451607465744019,
            'breasts': 0.9867608547210693, 'looking_at_viewer': 0.9200493693351746, 'blush': 0.8876285552978516,
            'smile': 0.5031097531318665, 'bangs': 0.4979058504104614, 'large_breasts': 0.5059964656829834,
            'navel': 0.9681310653686523, 'hair_between_eyes': 0.5816333293914795, 'medium_breasts': 0.36410677433013916,
            'very_long_hair': 0.811715304851532, 'closed_mouth': 0.9338403940200806, 'nipples': 0.9715133905410767,
            'purple_eyes': 0.9681202173233032, 'collarbone': 0.573296308517456, 'nude': 0.9568941593170166,
            'red_hair': 0.9242303967475891, 'sweat': 0.8757796287536621, 'horns': 0.973071277141571,
            'pussy': 0.9876313805580139, 'spread_legs': 0.9634276628494263, 'armpits': 0.9116500616073608,
            'stomach': 0.6858262419700623, 'arms_up': 0.9398491978645325, 'completely_nude': 0.907513439655304,
            'uncensored': 0.8703584671020508, 'pussy_juice': 0.6459053754806519,
            'feet_out_of_frame': 0.3921701908111572, 'on_bed': 0.6049470901489258,
            'arms_behind_head': 0.4758358597755432, 'breasts_apart': 0.38581883907318115,
            'clitoris': 0.5746099948883057
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9942929744720459}, abs=2e-2)

    def test_wd14_tags_sample_no_underline(self):
        rating, tags, chars = get_wd14_tags(get_testfile('nude_girl.png'), no_underline=True)
        assert rating == pytest.approx({
            'general': 0.0020540356636047363,
            'sensitive': 0.0080718994140625,
            'questionable': 0.003170192241668701,
            'explicit': 0.984081506729126,
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.998561441898346, 'solo': 0.9918843507766724, 'long hair': 0.9451607465744019,
            'breasts': 0.9867608547210693, 'looking at viewer': 0.9200493693351746, 'blush': 0.8876285552978516,
            'smile': 0.5031097531318665, 'bangs': 0.4979058504104614, 'large breasts': 0.5059964656829834,
            'navel': 0.9681310653686523, 'hair between eyes': 0.5816333293914795, 'medium breasts': 0.36410677433013916,
            'very long hair': 0.811715304851532, 'closed mouth': 0.9338403940200806, 'nipples': 0.9715133905410767,
            'purple eyes': 0.9681202173233032, 'collarbone': 0.573296308517456, 'nude': 0.9568941593170166,
            'red hair': 0.9242303967475891, 'sweat': 0.8757796287536621, 'horns': 0.973071277141571,
            'pussy': 0.9876313805580139, 'spread legs': 0.9634276628494263, 'armpits': 0.9116500616073608,
            'stomach': 0.6858262419700623, 'arms up': 0.9398491978645325, 'completely nude': 0.907513439655304,
            'uncensored': 0.8703584671020508, 'pussy juice': 0.6459053754806519,
            'feet out of frame': 0.3921701908111572, 'on bed': 0.6049470901489258,
            'arms behind head': 0.4758358597755432, 'breasts apart': 0.38581883907318115,
            'clitoris': 0.5746099948883057
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr (arknights)': 0.9942929744720459}, abs=2e-2)

    def test_wd14_tags_sample_mcut(self):
        rating, tags, chars = get_wd14_tags(
            get_testfile('nude_girl.png'),
            general_mcut_enabled=True,
            character_mcut_enabled=True,
        )
        assert rating == pytest.approx({
            'general': 0.0020540356636047363,
            'sensitive': 0.0080718994140625,
            'questionable': 0.003170192241668701,
            'explicit': 0.984081506729126,
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.998561441898346, 'solo': 0.9918843507766724, 'long_hair': 0.9451607465744019,
            'breasts': 0.9867608547210693, 'looking_at_viewer': 0.9200493693351746, 'blush': 0.8876285552978516,
            'navel': 0.9681310653686523, 'very_long_hair': 0.811715304851532, 'closed_mouth': 0.9338403940200806,
            'nipples': 0.9715133905410767, 'purple_eyes': 0.9681202173233032, 'nude': 0.9568941593170166,
            'red_hair': 0.9242303967475891, 'sweat': 0.8757796287536621, 'horns': 0.973071277141571,
            'pussy': 0.9876313805580139, 'spread_legs': 0.9634276628494263, 'armpits': 0.9116500616073608,
            'arms_up': 0.9398491978645325, 'completely_nude': 0.907513439655304, 'uncensored': 0.8703584671020508
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9942929744720459}, abs=2e-2)

    def test_wd14_tags_no_overlap(self):
        rating, tags, chars = get_wd14_tags(get_testfile('nude_girl.png'), drop_overlap=True)
        # print(tags)
        assert rating == pytest.approx({
            'general': 0.0020540356636047363,
            'sensitive': 0.0080718994140625,
            'questionable': 0.003170192241668701,
            'explicit': 0.984081506729126,
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.998561441898346, 'solo': 0.9918843507766724, 'looking_at_viewer': 0.9200493693351746,
            'blush': 0.8876285552978516, 'smile': 0.5031097531318665, 'bangs': 0.4979058504104614,
            'large_breasts': 0.5059964656829834, 'navel': 0.9681310653686523, 'hair_between_eyes': 0.5816333293914795,
            'medium_breasts': 0.36410677433013916, 'very_long_hair': 0.811715304851532,
            'closed_mouth': 0.9338403940200806, 'nipples': 0.9715133905410767, 'purple_eyes': 0.9681202173233032,
            'collarbone': 0.573296308517456, 'red_hair': 0.9242303967475891, 'sweat': 0.8757796287536621,
            'horns': 0.973071277141571, 'spread_legs': 0.9634276628494263, 'armpits': 0.9116500616073608,
            'stomach': 0.6858262419700623, 'arms_up': 0.9398491978645325, 'completely_nude': 0.907513439655304,
            'uncensored': 0.8703584671020508, 'pussy_juice': 0.6459053754806519,
            'feet_out_of_frame': 0.3921701908111572, 'on_bed': 0.6049470901489258,
            'arms_behind_head': 0.4758358597755432, 'breasts_apart': 0.38581883907318115,
            'clitoris': 0.5746099948883057
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9942929744720459}, abs=2e-2)
