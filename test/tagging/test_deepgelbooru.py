import pytest

from imgutils.tagging import get_deepgelbooru_tags
from imgutils.tagging.deepgelbooru import _open_tags, _open_model, _open_preprocessor
from test.testings import get_testfile


@pytest.fixture(autouse=True, scope='module')
def _release_model_after_run():
    try:
        yield
    finally:
        _open_tags.cache_clear()
        _open_model.cache_clear()
        _open_preprocessor.cache_clear()


@pytest.mark.unittest
class TestTaggingDeepgelbooru:
    def test_get_deepgelbooru_tags(self):
        rating, tags, chars = get_deepgelbooru_tags(get_testfile('6124220.jpg'))
        assert rating['rating:safe'] > 0.9
        assert tags['greyscale'] >= 0.5
        assert not chars

        rating, tags, chars = get_deepgelbooru_tags(get_testfile('6125785.jpg'))
        assert rating['rating:safe'] > 0.9
        assert tags['1girl'] >= 0.85
        assert tags['ring'] > 0.3
        assert chars['hu_tao_(genshin_impact)'] >= 0.7

    def test_get_gelbooru_tags_sample(self):
        rating, tags, chars = get_deepgelbooru_tags(get_testfile('nude_girl.png'))
        assert rating == pytest.approx({
            'rating:explicit': 0.9937806129455566,
            'rating:questionable': 0.0018548369407653809,
            'rating:safe': 0.0006630122661590576
        }, abs=1e-3)
        assert tags == pytest.approx({
            '1girl': 0.9938869476318359,
            'ahoge': 0.4203943908214569,
            'armpits': 0.9201775789260864,
            'arms_behind_head': 0.45194050669670105,
            'arms_up': 0.6920046806335449,
            'blush': 0.8007956743240356,
            'breasts': 0.9936412572860718,
            'closed_mouth': 0.4454791247844696,
            'collarbone': 0.4012811779975891,
            'completely_nude': 0.7057204246520996,
            'hair_between_eyes': 0.7024480700492859,
            'hair_intakes': 0.7307818531990051,
            'horns': 0.9652327299118042,
            'indoors': 0.31917262077331543,
            'long_hair': 0.9815766215324402,
            'looking_at_viewer': 0.9289669990539551,
            'medium_breasts': 0.7349288463592529,
            'navel': 0.9555190801620483,
            'nipples': 0.9295899271965027,
            'nude': 0.9620737433433533,
            'purple_eyes': 0.8718146085739136,
            'pussy': 0.988792896270752,
            'pussy_juice': 0.4703453481197357,
            'red_hair': 0.9628779292106628,
            'sitting': 0.5647267699241638,
            'smile': 0.6953961253166199,
            'solo': 0.9744712710380554,
            'spread_legs': 0.6513095498085022,
            'stomach': 0.6229314804077148,
            'sweat': 0.7899425029754639,
            'thighs': 0.6840589046478271,
            'uncensored': 0.826318621635437,
            'very_long_hair': 0.7851021885871887
        }, abs=1e-3)
        assert chars == pytest.approx({'surtr_(arknights)': 0.3173447847366333}, abs=1e-3)

    def test_get_gelbooru_tags_drop_overlap(self):
        rating, tags, chars = get_deepgelbooru_tags(get_testfile('nude_girl.png'), drop_overlap=True)
        assert rating == pytest.approx({
            'rating:explicit': 0.9937806129455566,
            'rating:questionable': 0.0018548369407653809,
            'rating:safe': 0.0006630122661590576
        }, abs=1e-3)
        assert tags == pytest.approx({
            '1girl': 0.9938869476318359,
            'ahoge': 0.4203943908214569,
            'armpits': 0.9201775789260864,
            'arms_behind_head': 0.45194050669670105,
            'arms_up': 0.6920046806335449,
            'blush': 0.8007956743240356,
            'closed_mouth': 0.4454791247844696,
            'collarbone': 0.4012811779975891,
            'completely_nude': 0.7057204246520996,
            'hair_between_eyes': 0.7024480700492859,
            'hair_intakes': 0.7307818531990051,
            'horns': 0.9652327299118042,
            'indoors': 0.31917262077331543,
            'looking_at_viewer': 0.9289669990539551,
            'medium_breasts': 0.7349288463592529,
            'navel': 0.9555190801620483,
            'nipples': 0.9295899271965027,
            'purple_eyes': 0.8718146085739136,
            'pussy_juice': 0.4703453481197357,
            'red_hair': 0.9628779292106628,
            'sitting': 0.5647267699241638,
            'smile': 0.6953961253166199,
            'solo': 0.9744712710380554,
            'spread_legs': 0.6513095498085022,
            'stomach': 0.6229314804077148,
            'sweat': 0.7899425029754639,
            'thighs': 0.6840589046478271,
            'uncensored': 0.826318621635437,
            'very_long_hair': 0.7851021885871887
        }, abs=1e-3)
        assert chars == pytest.approx({'surtr_(arknights)': 0.3173447847366333}, abs=1e-3)
