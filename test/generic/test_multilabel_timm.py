import pytest

from imgutils.generic import multilabel_timm_predict
from imgutils.generic.multilabel_timm import _open_models_for_repo_id
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestGenericMultilabelTIMM:
    def test_multilabel_timm_predict(self):
        general, character, rating = multilabel_timm_predict(
            get_testfile('nude_girl.png'),
            repo_id='animetimm/mobilenetv3_large_150d.dbv4-full',
            fmt=('general', 'character', 'rating'),
        )
        assert general == pytest.approx({
            '1girl': 0.9911611676216125, 'breasts': 0.9696003794670105, 'solo': 0.9610683917999268,
            'pussy': 0.960993766784668, 'nipples': 0.9577178955078125, 'horns': 0.9487239122390747,
            'long_hair': 0.9340348243713379, 'nude': 0.9182796478271484, 'purple_eyes': 0.90740966796875,
            'completely_nude': 0.8705511689186096, 'red_hair': 0.8630707263946533, 'navel': 0.8418680429458618,
            'uncensored': 0.8355356454849243, 'looking_at_viewer': 0.8342769145965576,
            'spread_legs': 0.7978109121322632, 'blush': 0.7877253293991089, 'smile': 0.6634377837181091,
            'armpits': 0.6523389220237732, 'stomach': 0.6226951479911804, 'very_long_hair': 0.621985137462616,
            'anus': 0.5988802313804626, 'hair_between_eyes': 0.5209077596664429, 'closed_mouth': 0.5168408155441284,
            'medium_breasts': 0.4368951916694641, 'arms_up': 0.4188764989376068, 'hair_intakes': 0.394428551197052,
            'thighs': 0.33927807211875916, 'cleft_of_venus': 0.31293246150016785, 'collarbone': 0.30162444710731506,
            'arms_behind_head': 0.2740631699562073, 'clitoris': 0.1854049563407898
        }, abs=1e-2)
        assert character == pytest.approx({'surtr_(arknights)': 0.9033797979354858}, abs=1e-2)
        assert rating == pytest.approx({'explicit': 0.9496212005615234}, abs=1e-2)

    def test_multilabel_timm_predict_noname(self):
        general, character, rating = multilabel_timm_predict(
            get_testfile('nude_girl.png'),
            repo_id='animetimm/mobilenetv3_large_150d.dbv4-full',
        )
        assert general == pytest.approx({
            '1girl': 0.9911611676216125, 'breasts': 0.9696003794670105, 'solo': 0.9610683917999268,
            'pussy': 0.960993766784668, 'nipples': 0.9577178955078125, 'horns': 0.9487239122390747,
            'long_hair': 0.9340348243713379, 'nude': 0.9182796478271484, 'purple_eyes': 0.90740966796875,
            'completely_nude': 0.8705511689186096, 'red_hair': 0.8630707263946533, 'navel': 0.8418680429458618,
            'uncensored': 0.8355356454849243, 'looking_at_viewer': 0.8342769145965576,
            'spread_legs': 0.7978109121322632, 'blush': 0.7877253293991089, 'smile': 0.6634377837181091,
            'armpits': 0.6523389220237732, 'stomach': 0.6226951479911804, 'very_long_hair': 0.621985137462616,
            'anus': 0.5988802313804626, 'hair_between_eyes': 0.5209077596664429, 'closed_mouth': 0.5168408155441284,
            'medium_breasts': 0.4368951916694641, 'arms_up': 0.4188764989376068, 'hair_intakes': 0.394428551197052,
            'thighs': 0.33927807211875916, 'cleft_of_venus': 0.31293246150016785, 'collarbone': 0.30162444710731506,
            'arms_behind_head': 0.2740631699562073, 'clitoris': 0.1854049563407898
        }, abs=1e-2)
        assert character == pytest.approx({'surtr_(arknights)': 0.9033797979354858}, abs=1e-2)
        assert rating == pytest.approx({'explicit': 0.9496212005615234}, abs=1e-2)

    def test_multilabel_timm_predict_with_category_thresholds(self):
        general, character, rating = multilabel_timm_predict(
            get_testfile('nude_girl.png'),
            repo_id='animetimm/mobilenetv3_large_150d.dbv4-full',
            thresholds={'general': 0.3},
            fmt=('general', 'character', 'rating'),
        )
        assert general == pytest.approx({
            '1girl': 0.9911611676216125, 'breasts': 0.9696003794670105, 'solo': 0.9610683917999268,
            'pussy': 0.960993766784668, 'nipples': 0.9577178955078125, 'horns': 0.9487239122390747,
            'long_hair': 0.9340348243713379, 'nude': 0.9182796478271484, 'purple_eyes': 0.90740966796875,
            'completely_nude': 0.8705511689186096, 'red_hair': 0.8630707263946533, 'navel': 0.8418680429458618,
            'uncensored': 0.8355356454849243, 'looking_at_viewer': 0.8342769145965576,
            'spread_legs': 0.7978109121322632, 'blush': 0.7877253293991089, 'smile': 0.6634377837181091,
            'armpits': 0.6523389220237732, 'stomach': 0.6226951479911804, 'very_long_hair': 0.621985137462616,
            'anus': 0.5988802313804626, 'hair_between_eyes': 0.5209077596664429, 'closed_mouth': 0.5168408155441284,
            'medium_breasts': 0.4368951916694641, 'arms_up': 0.4188764989376068, 'hair_intakes': 0.394428551197052,
            'thighs': 0.33927807211875916, 'cleft_of_venus': 0.31293246150016785, 'collarbone': 0.30162444710731506
        }, abs=1e-2)
        assert character == pytest.approx({'surtr_(arknights)': 0.9033797979354858}, abs=1e-2)
        assert rating == pytest.approx({'explicit': 0.9496212005615234}, abs=1e-2)

    def test_multilabel_timm_predict_with_category_thresholds_cateid(self):
        general, character, rating = multilabel_timm_predict(
            get_testfile('nude_girl.png'),
            repo_id='animetimm/mobilenetv3_large_150d.dbv4-full',
            thresholds={0: 0.3},
            fmt=('general', 'character', 'rating'),
        )
        assert general == pytest.approx({
            '1girl': 0.9911611676216125, 'breasts': 0.9696003794670105, 'solo': 0.9610683917999268,
            'pussy': 0.960993766784668, 'nipples': 0.9577178955078125, 'horns': 0.9487239122390747,
            'long_hair': 0.9340348243713379, 'nude': 0.9182796478271484, 'purple_eyes': 0.90740966796875,
            'completely_nude': 0.8705511689186096, 'red_hair': 0.8630707263946533, 'navel': 0.8418680429458618,
            'uncensored': 0.8355356454849243, 'looking_at_viewer': 0.8342769145965576,
            'spread_legs': 0.7978109121322632, 'blush': 0.7877253293991089, 'smile': 0.6634377837181091,
            'armpits': 0.6523389220237732, 'stomach': 0.6226951479911804, 'very_long_hair': 0.621985137462616,
            'anus': 0.5988802313804626, 'hair_between_eyes': 0.5209077596664429, 'closed_mouth': 0.5168408155441284,
            'medium_breasts': 0.4368951916694641, 'arms_up': 0.4188764989376068, 'hair_intakes': 0.394428551197052,
            'thighs': 0.33927807211875916, 'cleft_of_venus': 0.31293246150016785, 'collarbone': 0.30162444710731506
        }, abs=1e-2)
        assert character == pytest.approx({'surtr_(arknights)': 0.9033797979354858}, abs=1e-2)
        assert rating == pytest.approx({'explicit': 0.9496212005615234}, abs=1e-2)

    def test_multilabel_timm_predict_with_category_thresholds_cateid_generic_thresholds(self):
        general, character, rating = multilabel_timm_predict(
            get_testfile('nude_girl.png'),
            repo_id='animetimm/mobilenetv3_large_150d.dbv4-full',
            thresholds=0.3,
            fmt=('general', 'character', 'rating'),
        )
        assert general == pytest.approx({
            '1girl': 0.9911611676216125, 'breasts': 0.9696003794670105, 'solo': 0.9610683917999268,
            'pussy': 0.960993766784668, 'nipples': 0.9577178955078125, 'horns': 0.9487239122390747,
            'long_hair': 0.9340348243713379, 'nude': 0.9182796478271484, 'purple_eyes': 0.90740966796875,
            'completely_nude': 0.8705511689186096, 'red_hair': 0.8630707263946533, 'navel': 0.8418680429458618,
            'uncensored': 0.8355356454849243, 'looking_at_viewer': 0.8342769145965576,
            'spread_legs': 0.7978109121322632, 'blush': 0.7877253293991089, 'smile': 0.6634377837181091,
            'armpits': 0.6523389220237732, 'stomach': 0.6226951479911804, 'very_long_hair': 0.621985137462616,
            'anus': 0.5988802313804626, 'hair_between_eyes': 0.5209077596664429, 'closed_mouth': 0.5168408155441284,
            'medium_breasts': 0.4368951916694641, 'arms_up': 0.4188764989376068, 'hair_intakes': 0.394428551197052,
            'thighs': 0.33927807211875916, 'cleft_of_venus': 0.31293246150016785, 'collarbone': 0.30162444710731506
        }, abs=1e-2)
        assert character == pytest.approx({'surtr_(arknights)': 0.9033797979354858}, abs=1e-2)
        assert rating == pytest.approx({'explicit': 0.9496212005615234}, abs=1e-2)

    def test_multilabel_timm_predict_with_category_thresholds_cateid_no_tagth(self):
        general, character, rating = multilabel_timm_predict(
            get_testfile('nude_girl.png'),
            repo_id='animetimm/mobilenetv3_large_150d.dbv4-full',
            thresholds={0: 0.3},
            fmt=('general', 'character', 'rating'),
            use_tag_thresholds=False,
        )
        assert general == pytest.approx({
            '1girl': 0.9911611676216125, 'breasts': 0.9696003794670105, 'solo': 0.9610683917999268,
            'pussy': 0.960993766784668, 'nipples': 0.9577178955078125, 'horns': 0.9487239122390747,
            'long_hair': 0.9340348243713379, 'nude': 0.9182796478271484, 'purple_eyes': 0.90740966796875,
            'completely_nude': 0.8705511689186096, 'red_hair': 0.8630707263946533, 'navel': 0.8418680429458618,
            'uncensored': 0.8355356454849243, 'looking_at_viewer': 0.8342769145965576,
            'spread_legs': 0.7978109121322632, 'blush': 0.7877253293991089, 'smile': 0.6634377837181091,
            'armpits': 0.6523389220237732, 'stomach': 0.6226951479911804, 'very_long_hair': 0.621985137462616,
            'anus': 0.5988802313804626, 'hair_between_eyes': 0.5209077596664429, 'closed_mouth': 0.5168408155441284,
            'medium_breasts': 0.4368951916694641, 'arms_up': 0.4188764989376068, 'hair_intakes': 0.394428551197052,
            'thighs': 0.33927807211875916, 'cleft_of_venus': 0.31293246150016785, 'collarbone': 0.30162444710731506
        }, abs=1e-2)
        assert character == pytest.approx({'surtr_(arknights)': 0.9033797979354858}, abs=1e-2)
        assert rating == pytest.approx({'explicit': 0.9496212005615234}, abs=1e-2)

    def test_multilabel_timm_predict_with_category_thresholds_cateid_no_tagth_evalx(self):
        general, character, rating = multilabel_timm_predict(
            get_testfile('nude_girl.png'),
            repo_id='animetimm/mobilenetv3_large_150d.dbv4-full',
            thresholds={0: 0.3},
            fmt=('general', 'character', 'rating'),
            use_tag_thresholds=False,
            preprocessor='val',
        )
        assert general == pytest.approx({
            '1girl': 0.9911611676216125, 'breasts': 0.9696003794670105, 'solo': 0.9610683917999268,
            'pussy': 0.960993766784668, 'nipples': 0.9577178955078125, 'horns': 0.9487239122390747,
            'long_hair': 0.9340348243713379, 'nude': 0.9182796478271484, 'purple_eyes': 0.90740966796875,
            'completely_nude': 0.8705511689186096, 'red_hair': 0.8630707263946533, 'navel': 0.8418680429458618,
            'uncensored': 0.8355356454849243, 'looking_at_viewer': 0.8342769145965576,
            'spread_legs': 0.7978109121322632, 'blush': 0.7877253293991089, 'smile': 0.6634377837181091,
            'armpits': 0.6523389220237732, 'stomach': 0.6226951479911804, 'very_long_hair': 0.621985137462616,
            'anus': 0.5988802313804626, 'hair_between_eyes': 0.5209077596664429, 'closed_mouth': 0.5168408155441284,
            'medium_breasts': 0.4368951916694641, 'arms_up': 0.4188764989376068, 'hair_intakes': 0.394428551197052,
            'thighs': 0.33927807211875916, 'cleft_of_venus': 0.31293246150016785, 'collarbone': 0.30162444710731506
        }, abs=1e-2)
        assert character == pytest.approx({'surtr_(arknights)': 0.9033797979354858}, abs=1e-2)
        assert rating == pytest.approx({'explicit': 0.9496212005615234}, abs=1e-2)
