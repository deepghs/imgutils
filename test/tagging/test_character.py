import pytest

from imgutils.tagging import is_basic_character_tag, drop_basic_character_tags


@pytest.mark.unittest
class TestTaggingCharacter:
    def test_is_basic_character_tag(self):
        assert is_basic_character_tag('red_hair')
        assert is_basic_character_tag('red hair')
        assert is_basic_character_tag('blue eyes')
        assert is_basic_character_tag('blue eye')
        assert is_basic_character_tag('sheep ear')
        assert is_basic_character_tag('pointy ears')
        assert is_basic_character_tag('hair bun')

        assert is_basic_character_tag('hair over heads')
        assert is_basic_character_tag('hair_between_breasts')

        assert not is_basic_character_tag('chair')
        assert not is_basic_character_tag('hear')
        assert not is_basic_character_tag('drill')
        assert not is_basic_character_tag('drills')
        assert not is_basic_character_tag('pubic hair')

    def test_drop_basic_character_tags(self, complex_dict_tags, complex_list_tags):
        assert drop_basic_character_tags(complex_dict_tags) == pytest.approx({
            '1girl': 0.998362123966217, 'solo': 0.9912548065185547,
            'looking_at_viewer': 0.9146994352340698, 'blush': 0.8892400860786438, 'smile': 0.43393653631210327,
            'navel': 0.9653235077857971, 'closed_mouth': 0.9369247555732727,
            'nipples': 0.9660118222236633, 'collarbone': 0.588348925113678, 'nude': 0.9496222734451294,
            'sweat': 0.8690457344055176, 'pussy': 0.9868264198303223, 'spread_legs': 0.9603149890899658,
            'armpits': 0.9024748802185059, 'stomach': 0.6723923087120056, 'arms_up': 0.9380699396133423,
            'completely_nude': 0.9002960920333862, 'uncensored': 0.8612104058265686, 'pussy_juice': 0.6021570563316345,
            'feet_out_of_frame': 0.39779460430145264, 'on_bed': 0.610720157623291,
            'arms_behind_head': 0.44814401865005493, 'breasts_apart': 0.39798974990844727,
            'clitoris': 0.5310801267623901
        }, abs=1e-3)
        assert drop_basic_character_tags(complex_list_tags) == [
            '1girl', 'solo', 'looking_at_viewer', 'blush', 'smile', 'navel', 'closed_mouth',
            'nipples', 'collarbone', 'nude', 'sweat', 'pussy', 'spread_legs', 'armpits', 'stomach', 'arms_up',
            'completely_nude', 'uncensored', 'pussy_juice', 'feet_out_of_frame', 'on_bed', 'arms_behind_head',
            'breasts_apart', 'clitoris'
        ]

    def test_drop_basic_character_tags_error(self):
        with pytest.raises(TypeError):
            drop_basic_character_tags(122)
        with pytest.raises(TypeError):
            drop_basic_character_tags(None)
