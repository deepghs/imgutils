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

        rating, tags, chars = get_wd14_tags(get_testfile('6125785.jpg'))
        assert 0.6 <= rating['general'] <= 0.8
        assert tags['1girl'] >= 0.95
        assert chars['hu_tao_(genshin_impact)'] >= 0.95

    def test_wd14_tags_sample(self):
        rating, tags, chars = get_wd14_tags(get_testfile('nude_girl.png'))

        assert rating == pytest.approx({
            'general': 0.0006683468818664551,
            'sensitive': 0.003294050693511963,
            'questionable': 0.0007482171058654785,
            'explicit': 0.9922184944152832
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.9994567036628723, 'solo': 0.9867788553237915, 'long_hair': 0.9705560207366943,
            'breasts': 0.9950063228607178, 'looking_at_viewer': 0.9309853315353394, 'blush': 0.9086592793464661,
            'smile': 0.7154737710952759, 'navel': 0.9606291055679321, 'hair_between_eyes': 0.4996751546859741,
            'closed_mouth': 0.7993873953819275, 'very_long_hair': 0.7326497435569763,
            'medium_breasts': 0.7169027924537659, 'nipples': 0.9904205799102783, 'purple_eyes': 0.9592539668083191,
            'thighs': 0.4377020001411438, 'sweat': 0.6950557827949524, 'red_hair': 0.9731366038322449,
            'nude': 0.9811137318611145, 'lying': 0.5896710157394409, 'horns': 0.9796154499053955,
            'pussy': 0.9834838509559631, 'spread_legs': 0.9527802467346191, 'stomach': 0.7781887054443359,
            'on_back': 0.5563216805458069, 'armpits': 0.9518307447433472, 'arms_up': 0.8266783952713013,
            'completely_nude': 0.9168736338615417, 'pillow': 0.5372565388679504, 'uncensored': 0.9515247344970703,
            'pussy_juice': 0.6543970108032227, 'on_bed': 0.6051450371742249, 'hair_intakes': 0.8222305178642273,
            'demon_horns': 0.4169325828552246, 'breasts_apart': 0.45593249797821045, 'clitoris': 0.4843749403953552
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9940401315689087}, abs=2e-2)

    def test_wd14_tags_sample_no_underline(self):
        rating, tags, chars = get_wd14_tags(get_testfile('nude_girl.png'), no_underline=True)
        assert rating == pytest.approx({
            'general': 0.0006683468818664551,
            'sensitive': 0.003294050693511963,
            'questionable': 0.0007482171058654785,
            'explicit': 0.9922184944152832
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.9994567036628723, 'solo': 0.9867788553237915, 'long hair': 0.9705560207366943,
            'breasts': 0.9950063228607178, 'looking at viewer': 0.9309853315353394, 'blush': 0.9086592793464661,
            'smile': 0.7154737710952759, 'navel': 0.9606291055679321, 'hair between eyes': 0.4996751546859741,
            'closed mouth': 0.7993873953819275, 'very long hair': 0.7326497435569763,
            'medium breasts': 0.7169027924537659, 'nipples': 0.9904205799102783, 'purple eyes': 0.9592539668083191,
            'thighs': 0.4377020001411438, 'sweat': 0.6950557827949524, 'red hair': 0.9731366038322449,
            'nude': 0.9811137318611145, 'lying': 0.5896710157394409, 'horns': 0.9796154499053955,
            'pussy': 0.9834838509559631, 'spread legs': 0.9527802467346191, 'stomach': 0.7781887054443359,
            'on back': 0.5563216805458069, 'armpits': 0.9518307447433472, 'arms up': 0.8266783952713013,
            'completely nude': 0.9168736338615417, 'pillow': 0.5372565388679504, 'uncensored': 0.9515247344970703,
            'pussy juice': 0.6543970108032227, 'on bed': 0.6051450371742249, 'hair intakes': 0.8222305178642273,
            'demon horns': 0.4169325828552246, 'breasts apart': 0.45593249797821045, 'clitoris': 0.4843749403953552
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr (arknights)': 0.9942929744720459}, abs=2e-2)

    def test_wd14_tags_sample_mcut(self):
        rating, tags, chars = get_wd14_tags(
            get_testfile('nude_girl.png'),
            general_mcut_enabled=True,
            character_mcut_enabled=True,
        )
        assert rating == pytest.approx({
            'general': 0.0006683468818664551,
            'sensitive': 0.003294050693511963,
            'questionable': 0.0007482171058654785,
            'explicit': 0.9922184944152832
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.9994567036628723, 'solo': 0.9867788553237915, 'long_hair': 0.9705560207366943,
            'breasts': 0.9950063228607178, 'looking_at_viewer': 0.9309853315353394, 'blush': 0.9086592793464661,
            'navel': 0.9606291055679321, 'nipples': 0.9904205799102783, 'purple_eyes': 0.9592539668083191,
            'red_hair': 0.9731366038322449, 'nude': 0.9811137318611145, 'horns': 0.9796154499053955,
            'pussy': 0.9834838509559631, 'spread_legs': 0.9527802467346191, 'armpits': 0.9518307447433472,
            'completely_nude': 0.9168736338615417, 'uncensored': 0.9515247344970703
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9940401315689087}, abs=2e-2)

    def test_wd14_tags_no_overlap(self):
        rating, tags, chars = get_wd14_tags(get_testfile('nude_girl.png'), drop_overlap=True)
        assert rating == pytest.approx({
            'general': 0.0006683468818664551,
            'sensitive': 0.003294050693511963,
            'questionable': 0.0007482171058654785,
            'explicit': 0.9922184944152832
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.9994567036628723, 'solo': 0.9867788553237915, 'looking_at_viewer': 0.9309853315353394,
            'blush': 0.9086592793464661, 'smile': 0.7154737710952759, 'navel': 0.9606291055679321,
            'hair_between_eyes': 0.4996751546859741, 'closed_mouth': 0.7993873953819275,
            'very_long_hair': 0.7326497435569763, 'medium_breasts': 0.7169027924537659, 'nipples': 0.9904205799102783,
            'purple_eyes': 0.9592539668083191, 'thighs': 0.4377020001411438, 'sweat': 0.6950557827949524,
            'red_hair': 0.9731366038322449, 'spread_legs': 0.9527802467346191, 'stomach': 0.7781887054443359,
            'on_back': 0.5563216805458069, 'armpits': 0.9518307447433472, 'arms_up': 0.8266783952713013,
            'completely_nude': 0.9168736338615417, 'pillow': 0.5372565388679504, 'uncensored': 0.9515247344970703,
            'pussy_juice': 0.6543970108032227, 'on_bed': 0.6051450371742249, 'hair_intakes': 0.8222306370735168,
            'demon_horns': 0.4169325828552246, 'breasts_apart': 0.45593249797821045, 'clitoris': 0.4843749403953552
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.9940401315689087}, abs=2e-2)
