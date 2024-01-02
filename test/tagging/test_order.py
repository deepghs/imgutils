import pytest

from imgutils.tagging import sort_tags


@pytest.mark.unittest
class TestTaggingOrder:
    def test_sort_tags(self, complex_dict_tags, complex_list_tags):
        assert sort_tags(complex_dict_tags) == [
            'solo', '1girl', 'pussy', 'breasts', 'horns', 'purple_eyes', 'nipples', 'navel', 'spread_legs', 'nude',
            'long_hair', 'arms_up', 'closed_mouth', 'red_hair', 'looking_at_viewer', 'armpits', 'completely_nude',
            'blush', 'sweat', 'uncensored', 'very_long_hair', 'stomach', 'on_bed', 'pussy_juice', 'collarbone',
            'hair_between_eyes', 'clitoris', 'large_breasts', 'bangs', 'arms_behind_head', 'smile', 'breasts_apart',
            'feet_out_of_frame'
        ]
        with pytest.raises(TypeError):
            sort_tags(complex_list_tags)

    def test_sort_tags_score(self, complex_dict_tags, complex_list_tags):
        assert sort_tags(complex_dict_tags, mode='score') == [
            'solo', '1girl', 'pussy', 'breasts', 'horns', 'purple_eyes', 'nipples', 'navel', 'spread_legs', 'nude',
            'long_hair', 'arms_up', 'closed_mouth', 'red_hair', 'looking_at_viewer', 'armpits', 'completely_nude',
            'blush', 'sweat', 'uncensored', 'very_long_hair', 'stomach', 'on_bed', 'pussy_juice', 'collarbone',
            'hair_between_eyes', 'clitoris', 'large_breasts', 'bangs', 'arms_behind_head', 'smile', 'breasts_apart',
            'feet_out_of_frame'
        ]
        with pytest.raises(TypeError):
            sort_tags(complex_list_tags, mode='score')

    def test_sort_tags_original(self, complex_dict_tags, complex_list_tags):
        assert sort_tags(complex_dict_tags, mode='original') == [
            'solo', '1girl', 'long_hair', 'breasts', 'looking_at_viewer', 'blush', 'smile', 'bangs', 'large_breasts',
            'navel', 'hair_between_eyes', 'very_long_hair', 'closed_mouth', 'nipples', 'purple_eyes', 'collarbone',
            'nude', 'red_hair', 'sweat', 'horns', 'pussy', 'spread_legs', 'armpits', 'stomach', 'arms_up',
            'completely_nude', 'uncensored', 'pussy_juice', 'feet_out_of_frame', 'on_bed', 'arms_behind_head',
            'breasts_apart', 'clitoris'
        ]
        assert sort_tags(complex_list_tags, mode='original') == [
            'solo', '1girl', 'long_hair', 'breasts', 'looking_at_viewer', 'blush', 'smile', 'bangs', 'large_breasts',
            'navel', 'hair_between_eyes', 'very_long_hair', 'closed_mouth', 'nipples', 'purple_eyes', 'collarbone',
            'nude', 'red_hair', 'sweat', 'horns', 'pussy', 'spread_legs', 'armpits', 'stomach', 'arms_up',
            'completely_nude', 'uncensored', 'pussy_juice', 'feet_out_of_frame', 'on_bed', 'arms_behind_head',
            'breasts_apart', 'clitoris'
        ]

    def test_sort_tags_shuffle(self, complex_dict_tags, complex_list_tags):
        tpls = []
        for _ in range(10):
            new_tpl = tuple(sort_tags(complex_dict_tags, mode='shuffle'))
            assert new_tpl[:2] == ('solo', '1girl')
            tpls.append(new_tpl)
        assert len(set(tpls)) >= 9

        tpls = []
        for _ in range(10):
            new_tpl = tuple(sort_tags(complex_list_tags, mode='shuffle'))
            assert new_tpl[:2] == ('solo', '1girl')
            tpls.append(new_tpl)
        assert len(set(tpls)) >= 9

    def test_sort_tags_invalid_mode(self, complex_dict_tags):
        with pytest.raises(ValueError):
            _ = sort_tags(complex_dict_tags, mode='invalid mode')
