import numpy as np
import pytest

from imgutils.tagging import get_camie_tags
from imgutils.tagging.camie import _get_camie_model, _get_camie_preprocessor, _get_camie_threshold, _get_camie_labels
from test.testings import get_testfile


@pytest.fixture(autouse=True, scope='module')
def _release_model_after_run():
    try:
        yield
    finally:
        _get_camie_model.cache_clear()
        _get_camie_labels.cache_clear()
        _get_camie_preprocessor.cache_clear()
        _get_camie_threshold.cache_clear()


@pytest.mark.unittest
class TestTaggingCamie:
    def test_get_camie_tags_simple(self):
        rating, tags, chars = get_camie_tags(get_testfile('nude_girl.png'))
        assert rating == pytest.approx({
            'general': 0.0021970272064208984,
            'sensitive': 0.01693376898765564,
            'questionable': 0.15642279386520386,
            'explicit': 0.8108681440353394
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.93951416015625, 'blush': 0.563869059085846, 'breasts': 0.8779534697532654,
            'collarbone': 0.3603898882865906, 'long_hair': 0.7679457664489746, 'looking_at_viewer': 0.5607095956802368,
            'red_hair': 0.6585916876792908, 'smile': 0.3169093728065491, 'solo': 0.7545191049575806,
            'thighs': 0.30556416511535645, 'very_long_hair': 0.5271034240722656, 'arm_up': 0.360746294260025,
            'armpits': 0.6459736824035645, 'medium_breasts': 0.46896451711654663, 'purple_eyes': 0.6832128763198853,
            'barefoot': 0.41755908727645874, 'completely_nude': 0.6303801536560059, 'hair_ornament': 0.3720449209213257,
            'nude': 0.6793189644813538, 'sitting': 0.3081115484237671, 'hair_between_eyes': 0.37542057037353516,
            'arm_behind_head': 0.31600916385650635, 'closed_mouth': 0.43588995933532715, 'indoors': 0.40764182806015015,
            'lying': 0.3660048246383667, 'spread_legs': 0.5909104347229004, 'censored': 0.3911146819591522,
            'pussy': 0.7643163204193115, 'nipples': 0.8150820732116699, 'sweat': 0.33766573667526245,
            'navel': 0.6170458793640137, 'uncensored': 0.524355411529541, 'pillow': 0.38039106130599976,
            'clitoris': 0.290987491607666, 'arms_up': 0.45759230852127075, 'on_back': 0.3977510929107666,
            'mosaic_censoring': 0.4024934768676758, 'on_bed': 0.521205484867096, 'pussy_juice': 0.32266202569007874,
            'hair_intakes': 0.41293373703956604, 'horns': 0.6854047775268555, 'bed_sheet': 0.29966840147972107,
            'breasts_apart': 0.38147661089897156, 'arms_behind_head': 0.29315847158432007, 'anus': 0.42645150423049927,
            'stomach': 0.4014701843261719, 'demon_horns': 0.3103789687156677
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.6217852830886841}, abs=2e-2)

    def test_get_camie_tags_all(self):
        rating, tags, chars, artists, copyrights, metas, years = get_camie_tags(
            get_testfile('nude_girl.png'),
            fmt=('rating', 'general', 'character', 'artist', 'copyright', 'meta', 'year')
        )
        assert rating == pytest.approx({
            'general': 0.0021970272064208984,
            'sensitive': 0.01693376898765564,
            'questionable': 0.15642279386520386,
            'explicit': 0.8108681440353394
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.93951416015625, 'blush': 0.563869059085846, 'breasts': 0.8779534697532654,
            'collarbone': 0.3603898882865906, 'long_hair': 0.7679457664489746, 'looking_at_viewer': 0.5607095956802368,
            'red_hair': 0.6585916876792908, 'smile': 0.3169093728065491, 'solo': 0.7545191049575806,
            'thighs': 0.30556416511535645, 'very_long_hair': 0.5271034240722656, 'arm_up': 0.360746294260025,
            'armpits': 0.6459736824035645, 'medium_breasts': 0.46896451711654663, 'purple_eyes': 0.6832128763198853,
            'barefoot': 0.41755908727645874, 'completely_nude': 0.6303801536560059, 'hair_ornament': 0.3720449209213257,
            'nude': 0.6793189644813538, 'sitting': 0.3081115484237671, 'hair_between_eyes': 0.37542057037353516,
            'arm_behind_head': 0.31600916385650635, 'closed_mouth': 0.43588995933532715, 'indoors': 0.40764182806015015,
            'lying': 0.3660048246383667, 'spread_legs': 0.5909104347229004, 'censored': 0.3911146819591522,
            'pussy': 0.7643163204193115, 'nipples': 0.8150820732116699, 'sweat': 0.33766573667526245,
            'navel': 0.6170458793640137, 'uncensored': 0.524355411529541, 'pillow': 0.38039106130599976,
            'clitoris': 0.290987491607666, 'arms_up': 0.45759230852127075, 'on_back': 0.3977510929107666,
            'mosaic_censoring': 0.4024934768676758, 'on_bed': 0.521205484867096, 'pussy_juice': 0.32266202569007874,
            'hair_intakes': 0.41293373703956604, 'horns': 0.6854047775268555, 'bed_sheet': 0.29966840147972107,
            'breasts_apart': 0.38147661089897156, 'arms_behind_head': 0.29315847158432007, 'anus': 0.42645150423049927,
            'stomach': 0.4014701843261719, 'demon_horns': 0.3103789687156677
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.6217852830886841}, abs=2e-2)
        assert artists == pytest.approx({}, abs=2e-2)
        assert copyrights == pytest.approx({'arknights': 0.8855481147766113}, abs=2e-2)
        assert metas == pytest.approx({'commentary': 0.3463279902935028, 'commentary_request': 0.26654714345932007,
                                       'paid_reward_available': 0.34057503938674927}, abs=2e-2)
        assert years == pytest.approx({'year_2021': 0.3289925158023834, 'year_2022': 0.370042622089386}, abs=2e-2)

    def test_get_camie_tags_with_data(self):
        rating, tags, chars, embedding, logits, prediction = get_camie_tags(
            get_testfile('nude_girl.png'),
            fmt=('rating', 'general', 'character', 'embedding', 'logits', 'prediction')
        )
        assert rating == pytest.approx({
            'general': 0.0021970272064208984,
            'sensitive': 0.01693376898765564,
            'questionable': 0.15642279386520386,
            'explicit': 0.8108681440353394
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.93951416015625, 'blush': 0.563869059085846, 'breasts': 0.8779534697532654,
            'collarbone': 0.3603898882865906, 'long_hair': 0.7679457664489746, 'looking_at_viewer': 0.5607095956802368,
            'red_hair': 0.6585916876792908, 'smile': 0.3169093728065491, 'solo': 0.7545191049575806,
            'thighs': 0.30556416511535645, 'very_long_hair': 0.5271034240722656, 'arm_up': 0.360746294260025,
            'armpits': 0.6459736824035645, 'medium_breasts': 0.46896451711654663, 'purple_eyes': 0.6832128763198853,
            'barefoot': 0.41755908727645874, 'completely_nude': 0.6303801536560059, 'hair_ornament': 0.3720449209213257,
            'nude': 0.6793189644813538, 'sitting': 0.3081115484237671, 'hair_between_eyes': 0.37542057037353516,
            'arm_behind_head': 0.31600916385650635, 'closed_mouth': 0.43588995933532715, 'indoors': 0.40764182806015015,
            'lying': 0.3660048246383667, 'spread_legs': 0.5909104347229004, 'censored': 0.3911146819591522,
            'pussy': 0.7643163204193115, 'nipples': 0.8150820732116699, 'sweat': 0.33766573667526245,
            'navel': 0.6170458793640137, 'uncensored': 0.524355411529541, 'pillow': 0.38039106130599976,
            'clitoris': 0.290987491607666, 'arms_up': 0.45759230852127075, 'on_back': 0.3977510929107666,
            'mosaic_censoring': 0.4024934768676758, 'on_bed': 0.521205484867096, 'pussy_juice': 0.32266202569007874,
            'hair_intakes': 0.41293373703956604, 'horns': 0.6854047775268555, 'bed_sheet': 0.29966840147972107,
            'breasts_apart': 0.38147661089897156, 'arms_behind_head': 0.29315847158432007, 'anus': 0.42645150423049927,
            'stomach': 0.4014701843261719, 'demon_horns': 0.3103789687156677
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.6217852830886841}, abs=2e-2)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1280,)
        assert isinstance(logits, np.ndarray)
        assert logits.shape == (70527,)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (70527,)

    def test_get_camie_tags_all_with_global_threshold(self):
        rating, tags, chars, artists, copyrights, metas, years = get_camie_tags(
            get_testfile('nude_girl.png'),
            fmt=('rating', 'general', 'character', 'artist', 'copyright', 'meta', 'year'),
            thresholds=0.5,
        )
        assert rating == pytest.approx({
            'general': 0.0021970272064208984,
            'sensitive': 0.01693376898765564,
            'questionable': 0.15642279386520386,
            'explicit': 0.8108681440353394
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.93951416015625, 'blush': 0.563869059085846, 'breasts': 0.8779534697532654,
            'long_hair': 0.7679457664489746, 'looking_at_viewer': 0.5607095956802368, 'red_hair': 0.6585916876792908,
            'solo': 0.7545191049575806, 'very_long_hair': 0.5271034240722656, 'armpits': 0.6459736824035645,
            'purple_eyes': 0.6832128763198853, 'completely_nude': 0.6303801536560059, 'nude': 0.6793189644813538,
            'spread_legs': 0.5909104347229004, 'pussy': 0.7643163204193115, 'nipples': 0.8150820732116699,
            'navel': 0.6170458793640137, 'uncensored': 0.524355411529541, 'on_bed': 0.521205484867096,
            'horns': 0.6854047775268555
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.6217852830886841}, abs=2e-2)
        assert artists == pytest.approx({}, abs=2e-2)
        assert copyrights == pytest.approx({'arknights': 0.8855481147766113}, abs=2e-2)
        assert metas == pytest.approx({}, abs=2e-2)
        assert years == pytest.approx({}, abs=2e-2)

    def test_get_camie_tags_all_with_partial_threshold(self):
        rating, tags, chars, artists, copyrights, metas, years = get_camie_tags(
            get_testfile('nude_girl.png'),
            fmt=('rating', 'general', 'character', 'artist', 'copyright', 'meta', 'year'),
            thresholds={'general': 0.5},
        )
        assert rating == pytest.approx({
            'general': 0.0021970272064208984,
            'sensitive': 0.01693376898765564,
            'questionable': 0.15642279386520386,
            'explicit': 0.8108681440353394
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.93951416015625, 'blush': 0.563869059085846, 'breasts': 0.8779534697532654,
            'long_hair': 0.7679457664489746, 'looking_at_viewer': 0.5607095956802368, 'red_hair': 0.6585916876792908,
            'solo': 0.7545191049575806, 'very_long_hair': 0.5271034240722656, 'armpits': 0.6459736824035645,
            'purple_eyes': 0.6832128763198853, 'completely_nude': 0.6303801536560059, 'nude': 0.6793189644813538,
            'spread_legs': 0.5909104347229004, 'pussy': 0.7643163204193115, 'nipples': 0.8150820732116699,
            'navel': 0.6170458793640137, 'uncensored': 0.524355411529541, 'on_bed': 0.521205484867096,
            'horns': 0.6854047775268555
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.6217852830886841}, abs=2e-2)
        assert artists == pytest.approx({}, abs=2e-2)
        assert copyrights == pytest.approx({'arknights': 0.8855481147766113}, abs=2e-2)
        assert metas == pytest.approx({'commentary': 0.3463279902935028, 'commentary_request': 0.26654714345932007,
                                       'paid_reward_available': 0.34057503938674927}, abs=2e-2)
        assert years == pytest.approx({'year_2021': 0.3289925158023834, 'year_2022': 0.370042622089386}, abs=2e-2)

    def test_get_camie_tags_refined(self):
        rating, tags, chars = get_camie_tags(
            get_testfile('nude_girl.png'),
            model_name='refined',
        )
        assert rating == pytest.approx({
            'general': 0.027370810508728027,
            'sensitive': 0.20533186197280884,
            'questionable': 0.44456690549850464,
            'explicit': 0.5236345529556274
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.9787406325340271, 'blush': 0.46126890182495117, 'breasts': 0.7486366629600525,
            'collarbone': 0.30797243118286133, 'long_hair': 0.662933886051178, 'looking_at_viewer': 0.487204372882843,
            'red_hair': 0.32015570998191833, 'smile': 0.3928636610507965, 'solo': 0.7213500142097473,
            'thighhighs': 0.3137112259864807, 'thighs': 0.33342698216438293, 'very_long_hair': 0.39193886518478394,
            'arm_up': 0.4579657316207886, 'armpits': 0.43291330337524414, 'medium_breasts': 0.4673271179199219,
            'purple_eyes': 0.4446716606616974, 'nude': 0.31779128313064575, 'hair_between_eyes': 0.3128393590450287,
            'short_sleeves': 0.27395501732826233, 'closed_mouth': 0.37338072061538696, 'lying': 0.3452643156051636,
            'spread_legs': 0.45434120297431946, 'dress': 0.35450559854507446, 'pussy': 0.45826300978660583,
            'blue_eyes': 0.2666358947753906, 'nipples': 0.456929475069046, 'sweat': 0.3484736680984497,
            'navel': 0.3848545253276825, 'arms_up': 0.3924522399902344, 'on_back': 0.31698185205459595
        }, abs=2e-2)
        assert chars == pytest.approx({}, abs=2e-2)

    def test_get_camie_tags_refined_with_data(self):
        rating, tags, chars, embedding, logits, prediction = get_camie_tags(
            get_testfile('nude_girl.png'),
            model_name='refined',
            fmt=('rating', 'general', 'character', 'embedding', 'logits', 'prediction')
        )
        assert rating == pytest.approx({
            'general': 0.027370810508728027,
            'sensitive': 0.20533186197280884,
            'questionable': 0.44456690549850464,
            'explicit': 0.5236345529556274
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.9787406325340271, 'blush': 0.46126890182495117, 'breasts': 0.7486366629600525,
            'collarbone': 0.30797243118286133, 'long_hair': 0.662933886051178, 'looking_at_viewer': 0.487204372882843,
            'red_hair': 0.32015570998191833, 'smile': 0.3928636610507965, 'solo': 0.7213500142097473,
            'thighhighs': 0.3137112259864807, 'thighs': 0.33342698216438293, 'very_long_hair': 0.39193886518478394,
            'arm_up': 0.4579657316207886, 'armpits': 0.43291330337524414, 'medium_breasts': 0.4673271179199219,
            'purple_eyes': 0.4446716606616974, 'nude': 0.31779128313064575, 'hair_between_eyes': 0.3128393590450287,
            'short_sleeves': 0.27395501732826233, 'closed_mouth': 0.37338072061538696, 'lying': 0.3452643156051636,
            'spread_legs': 0.45434120297431946, 'dress': 0.35450559854507446, 'pussy': 0.45826300978660583,
            'blue_eyes': 0.2666358947753906, 'nipples': 0.456929475069046, 'sweat': 0.3484736680984497,
            'navel': 0.3848545253276825, 'arms_up': 0.3924522399902344, 'on_back': 0.31698185205459595
        }, abs=2e-2)
        assert chars == pytest.approx({}, abs=2e-2)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (2560,)
        assert isinstance(logits, np.ndarray)
        assert logits.shape == (70527,)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (70527,)

    def test_get_camie_tags_refined_with_initial(self):
        rating, tags, chars, embedding, logits, prediction, \
            init_rating, init_tags, init_chars, init_embedding, init_logits, init_prediction = get_camie_tags(
            get_testfile('nude_girl.png'),
            model_name='refined',
            fmt=(
                'rating', 'general', 'character', 'embedding', 'logits', 'prediction',
                'initial/rating', 'initial/general', 'initial/character',
                'initial/embedding', 'initial/logits', 'initial/prediction'
            )
        )
        assert rating == pytest.approx({
            'general': 0.027370810508728027,
            'sensitive': 0.20533186197280884,
            'questionable': 0.44456690549850464,
            'explicit': 0.5236345529556274
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.9787406325340271, 'blush': 0.46126890182495117, 'breasts': 0.7486366629600525,
            'collarbone': 0.30797243118286133, 'long_hair': 0.662933886051178, 'looking_at_viewer': 0.487204372882843,
            'red_hair': 0.32015570998191833, 'smile': 0.3928636610507965, 'solo': 0.7213500142097473,
            'thighhighs': 0.3137112259864807, 'thighs': 0.33342698216438293, 'very_long_hair': 0.39193886518478394,
            'arm_up': 0.4579657316207886, 'armpits': 0.43291330337524414, 'medium_breasts': 0.4673271179199219,
            'purple_eyes': 0.4446716606616974, 'nude': 0.31779128313064575, 'hair_between_eyes': 0.3128393590450287,
            'short_sleeves': 0.27395501732826233, 'closed_mouth': 0.37338072061538696, 'lying': 0.3452643156051636,
            'spread_legs': 0.45434120297431946, 'dress': 0.35450559854507446, 'pussy': 0.45826300978660583,
            'blue_eyes': 0.2666358947753906, 'nipples': 0.456929475069046, 'sweat': 0.3484736680984497,
            'navel': 0.3848545253276825, 'arms_up': 0.3924522399902344, 'on_back': 0.31698185205459595
        }, abs=2e-2)
        assert chars == pytest.approx({}, abs=2e-2)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (2560,)
        assert isinstance(logits, np.ndarray)
        assert logits.shape == (70527,)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (70527,)

        assert init_rating == pytest.approx({
            'general': 0.0021970272064208984,
            'sensitive': 0.01693376898765564,
            'questionable': 0.15642279386520386,
            'explicit': 0.8108681440353394
        }, abs=2e-2)
        assert init_tags == pytest.approx({
            '1girl': 0.93951416015625, 'blush': 0.563869059085846, 'breasts': 0.8779534697532654,
            'collarbone': 0.3603898882865906, 'long_hair': 0.7679457664489746, 'looking_at_viewer': 0.5607095956802368,
            'red_hair': 0.6585916876792908, 'smile': 0.3169093728065491, 'solo': 0.7545191049575806,
            'thighs': 0.30556416511535645, 'very_long_hair': 0.5271034240722656, 'arm_up': 0.360746294260025,
            'armpits': 0.6459736824035645, 'medium_breasts': 0.46896451711654663, 'purple_eyes': 0.6832128763198853,
            'barefoot': 0.41755908727645874, 'completely_nude': 0.6303801536560059, 'hair_ornament': 0.3720449209213257,
            'nude': 0.6793189644813538, 'sitting': 0.3081115484237671, 'hair_between_eyes': 0.37542057037353516,
            'arm_behind_head': 0.31600916385650635, 'closed_mouth': 0.43588995933532715, 'indoors': 0.40764182806015015,
            'lying': 0.3660048246383667, 'spread_legs': 0.5909104347229004, 'censored': 0.3911146819591522,
            'pussy': 0.7643163204193115, 'nipples': 0.8150820732116699, 'sweat': 0.33766573667526245,
            'navel': 0.6170458793640137, 'uncensored': 0.524355411529541, 'pillow': 0.38039106130599976,
            'clitoris': 0.290987491607666, 'arms_up': 0.45759230852127075, 'on_back': 0.3977510929107666,
            'mosaic_censoring': 0.4024934768676758, 'on_bed': 0.521205484867096, 'pussy_juice': 0.32266202569007874,
            'hair_intakes': 0.41293373703956604, 'horns': 0.6854047775268555, 'bed_sheet': 0.29966840147972107,
            'breasts_apart': 0.38147661089897156, 'arms_behind_head': 0.29315847158432007, 'anus': 0.42645150423049927,
            'stomach': 0.4014701843261719, 'demon_horns': 0.3103789687156677
        }, abs=2e-2)
        assert init_chars == pytest.approx({'surtr_(arknights)': 0.6217852830886841}, abs=2e-2)

        assert isinstance(init_embedding, np.ndarray)
        assert init_embedding.shape == (1280,)
        assert isinstance(init_logits, np.ndarray)
        assert init_logits.shape == (70527,)
        assert isinstance(init_prediction, np.ndarray)
        assert init_prediction.shape == (70527,)

    def test_get_camie_tags_refined_initial_only(self):
        rating, tags, chars, embedding, logits, prediction = get_camie_tags(
            get_testfile('nude_girl.png'),
            model_name='refined',
            fmt=(
                'initial/rating', 'initial/general', 'initial/character',
                'initial/embedding', 'initial/logits', 'initial/prediction'
            )
        )
        assert rating == pytest.approx({
            'general': 0.0021970272064208984,
            'sensitive': 0.01693376898765564,
            'questionable': 0.15642279386520386,
            'explicit': 0.8108681440353394
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.93951416015625, 'blush': 0.563869059085846, 'breasts': 0.8779534697532654,
            'collarbone': 0.3603898882865906, 'long_hair': 0.7679457664489746, 'looking_at_viewer': 0.5607095956802368,
            'red_hair': 0.6585916876792908, 'smile': 0.3169093728065491, 'solo': 0.7545191049575806,
            'thighs': 0.30556416511535645, 'very_long_hair': 0.5271034240722656, 'arm_up': 0.360746294260025,
            'armpits': 0.6459736824035645, 'medium_breasts': 0.46896451711654663, 'purple_eyes': 0.6832128763198853,
            'barefoot': 0.41755908727645874, 'completely_nude': 0.6303801536560059, 'hair_ornament': 0.3720449209213257,
            'nude': 0.6793189644813538, 'sitting': 0.3081115484237671, 'hair_between_eyes': 0.37542057037353516,
            'arm_behind_head': 0.31600916385650635, 'closed_mouth': 0.43588995933532715, 'indoors': 0.40764182806015015,
            'lying': 0.3660048246383667, 'spread_legs': 0.5909104347229004, 'censored': 0.3911146819591522,
            'pussy': 0.7643163204193115, 'nipples': 0.8150820732116699, 'sweat': 0.33766573667526245,
            'navel': 0.6170458793640137, 'uncensored': 0.524355411529541, 'pillow': 0.38039106130599976,
            'clitoris': 0.290987491607666, 'arms_up': 0.45759230852127075, 'on_back': 0.3977510929107666,
            'mosaic_censoring': 0.4024934768676758, 'on_bed': 0.521205484867096, 'pussy_juice': 0.32266202569007874,
            'hair_intakes': 0.41293373703956604, 'horns': 0.6854047775268555, 'bed_sheet': 0.29966840147972107,
            'breasts_apart': 0.38147661089897156, 'arms_behind_head': 0.29315847158432007, 'anus': 0.42645150423049927,
            'stomach': 0.4014701843261719, 'demon_horns': 0.3103789687156677
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.6217852830886841}, abs=2e-2)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1280,)
        assert isinstance(logits, np.ndarray)
        assert logits.shape == (70527,)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (70527,)

    def test_get_camie_tags_simple_no_overlap_no_underline(self):
        rating, tags, chars = get_camie_tags(
            get_testfile('nude_girl.png'),
            no_underline=True,
            drop_overlap=True,
        )
        assert rating == pytest.approx({
            'general': 0.0021970272064208984,
            'sensitive': 0.01693376898765564,
            'questionable': 0.15642279386520386,
            'explicit': 0.8108681440353394
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.93951416015625, 'blush': 0.563869059085846, 'collarbone': 0.3603898882865906,
            'looking at viewer': 0.5607095956802368, 'red hair': 0.6585916876792908, 'smile': 0.3169093728065491,
            'solo': 0.7545191049575806, 'thighs': 0.30556416511535645, 'very long hair': 0.5271034240722656,
            'arm up': 0.360746294260025, 'armpits': 0.6459736824035645, 'medium breasts': 0.46896451711654663,
            'purple eyes': 0.6832128763198853, 'barefoot': 0.41755908727645874, 'completely nude': 0.6303801536560059,
            'hair ornament': 0.3720449209213257, 'sitting': 0.3081115484237671,
            'hair between eyes': 0.37542057037353516, 'arm behind head': 0.31600916385650635,
            'closed mouth': 0.43588995933532715, 'indoors': 0.40764182806015015, 'spread legs': 0.5909104347229004,
            'nipples': 0.8150820732116699, 'sweat': 0.33766573667526245, 'navel': 0.6170458793640137,
            'uncensored': 0.524355411529541, 'pillow': 0.38039106130599976, 'clitoris': 0.290987491607666,
            'arms up': 0.45759230852127075, 'on back': 0.3977510929107666, 'mosaic censoring': 0.4024934768676758,
            'on bed': 0.521205484867096, 'pussy juice': 0.32266202569007874, 'hair intakes': 0.41293373703956604,
            'bed sheet': 0.29966840147972107, 'breasts apart': 0.38147661089897156,
            'arms behind head': 0.29315847158432007, 'anus': 0.42645150423049927, 'stomach': 0.4014701843261719,
            'demon horns': 0.3103789687156677
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr (arknights)': 0.6217852830886841}, abs=2e-2)

    def test_invalid_threshold(self):
        with pytest.raises(TypeError):
            get_camie_tags(
                get_testfile('nude_girl.png'),
                thresholds='invalid type',
            )
