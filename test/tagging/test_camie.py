import numpy as np
import pytest

from imgutils.tagging import get_camie_tags
from imgutils.tagging.camie import _get_camie_model, _get_camie_preprocessor, _get_camie_threshold, _get_camie_labels, \
    convert_camie_emb_to_prediction, _get_camie_emb_to_pred_model
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
        _get_camie_emb_to_pred_model.cache_clear()


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

    def test_convert_initial(self):
        embedding = get_camie_tags(get_testfile('nude_girl.png'), fmt='embedding')
        rating, tags, chars = convert_camie_emb_to_prediction(embedding)
        assert rating == pytest.approx({
            'general': 0.0021970272064208984,
            'sensitive': 0.01693376898765564,
            'questionable': 0.15642279386520386,
            'explicit': 0.8108681440353394
        }, abs=2e-2)
        assert tags == pytest.approx({
            '1girl': 0.93951416015625, 'blush': 0.563869059085846, 'breasts': 0.8779534697532654,
            'collarbone': 0.3603898882865906, 'long_hair': 0.7679457664489746,
            'looking_at_viewer': 0.5607095956802368,
            'red_hair': 0.6585916876792908, 'smile': 0.3169093728065491, 'solo': 0.7545191049575806,
            'thighs': 0.30556416511535645, 'very_long_hair': 0.5271034240722656, 'arm_up': 0.360746294260025,
            'armpits': 0.6459736824035645, 'medium_breasts': 0.46896451711654663, 'purple_eyes': 0.6832128763198853,
            'barefoot': 0.41755908727645874, 'completely_nude': 0.6303801536560059,
            'hair_ornament': 0.3720449209213257,
            'nude': 0.6793189644813538, 'sitting': 0.3081115484237671, 'hair_between_eyes': 0.37542057037353516,
            'arm_behind_head': 0.31600916385650635, 'closed_mouth': 0.43588995933532715,
            'indoors': 0.40764182806015015,
            'lying': 0.3660048246383667, 'spread_legs': 0.5909104347229004, 'censored': 0.3911146819591522,
            'pussy': 0.7643163204193115, 'nipples': 0.8150820732116699, 'sweat': 0.33766573667526245,
            'navel': 0.6170458793640137, 'uncensored': 0.524355411529541, 'pillow': 0.38039106130599976,
            'clitoris': 0.290987491607666, 'arms_up': 0.45759230852127075, 'on_back': 0.3977510929107666,
            'mosaic_censoring': 0.4024934768676758, 'on_bed': 0.521205484867096, 'pussy_juice': 0.32266202569007874,
            'hair_intakes': 0.41293373703956604, 'horns': 0.6854047775268555, 'bed_sheet': 0.29966840147972107,
            'breasts_apart': 0.38147661089897156, 'arms_behind_head': 0.29315847158432007,
            'anus': 0.42645150423049927,
            'stomach': 0.4014701843261719, 'demon_horns': 0.3103789687156677
        }, abs=2e-2)
        assert chars == pytest.approx({'surtr_(arknights)': 0.6217852830886841}, abs=2e-2)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1280,)

    def test_convert_refined(self):
        embedding = get_camie_tags(get_testfile('nude_girl.png'), fmt='embedding', model_name='refined')
        rating, tags, chars = convert_camie_emb_to_prediction(embedding, model_name='refined')
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

    def test_convert_refined_s_initial(self):
        embedding = get_camie_tags(get_testfile('nude_girl.png'), fmt='initial/embedding', model_name='refined')
        rating, tags, chars = convert_camie_emb_to_prediction(embedding, model_name='refined', is_refined=False)
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

    def test_convert_initial_multiple(self):
        emb1 = get_camie_tags(get_testfile('nude_girl.png'), fmt='embedding')
        emb2 = get_camie_tags(get_testfile('skadi.jpg'), fmt='embedding')
        emb3 = get_camie_tags(get_testfile('hutao.jpg'), fmt='embedding')
        embedding = np.stack([emb1, emb2, emb3])
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (3, 1280)
        retval = convert_camie_emb_to_prediction(embedding)
        assert isinstance(retval, list)

        assert retval[0][0] == pytest.approx({
            'general': 0.0021970272064208984,
            'sensitive': 0.01693376898765564,
            'questionable': 0.15642279386520386,
            'explicit': 0.8108681440353394
        }, abs=2e-2)
        assert retval[0][1] == pytest.approx({
            '1girl': 0.93951416015625, 'blush': 0.563869059085846, 'breasts': 0.8779534697532654,
            'collarbone': 0.3603898882865906, 'long_hair': 0.7679457664489746,
            'looking_at_viewer': 0.5607095956802368,
            'red_hair': 0.6585916876792908, 'smile': 0.3169093728065491, 'solo': 0.7545191049575806,
            'thighs': 0.30556416511535645, 'very_long_hair': 0.5271034240722656, 'arm_up': 0.360746294260025,
            'armpits': 0.6459736824035645, 'medium_breasts': 0.46896451711654663, 'purple_eyes': 0.6832128763198853,
            'barefoot': 0.41755908727645874, 'completely_nude': 0.6303801536560059,
            'hair_ornament': 0.3720449209213257,
            'nude': 0.6793189644813538, 'sitting': 0.3081115484237671, 'hair_between_eyes': 0.37542057037353516,
            'arm_behind_head': 0.31600916385650635, 'closed_mouth': 0.43588995933532715,
            'indoors': 0.40764182806015015,
            'lying': 0.3660048246383667, 'spread_legs': 0.5909104347229004, 'censored': 0.3911146819591522,
            'pussy': 0.7643163204193115, 'nipples': 0.8150820732116699, 'sweat': 0.33766573667526245,
            'navel': 0.6170458793640137, 'uncensored': 0.524355411529541, 'pillow': 0.38039106130599976,
            'clitoris': 0.290987491607666, 'arms_up': 0.45759230852127075, 'on_back': 0.3977510929107666,
            'mosaic_censoring': 0.4024934768676758, 'on_bed': 0.521205484867096, 'pussy_juice': 0.32266202569007874,
            'hair_intakes': 0.41293373703956604, 'horns': 0.6854047775268555, 'bed_sheet': 0.29966840147972107,
            'breasts_apart': 0.38147661089897156, 'arms_behind_head': 0.29315847158432007,
            'anus': 0.42645150423049927,
            'stomach': 0.4014701843261719, 'demon_horns': 0.3103789687156677
        }, abs=2e-2)
        assert retval[0][2] == pytest.approx({'surtr_(arknights)': 0.6217852830886841}, abs=2e-2)

        assert retval[1][0] == pytest.approx({
            'general': 0.04246556758880615,
            'sensitive': 0.6936423778533936,
            'questionable': 0.23721203207969666,
            'explicit': 0.033293724060058594
        }, abs=2e-2)
        assert retval[1][1] == pytest.approx({
            '1girl': 0.8412569165229797, 'blush': 0.38029077649116516, 'breasts': 0.618192195892334,
            'cowboy_shot': 0.37446439266204834, 'large_breasts': 0.5698797702789307, 'long_hair': 0.7119565010070801,
            'looking_at_viewer': 0.5252856612205505, 'shirt': 0.46417444944381714, 'solo': 0.5428758859634399,
            'standing': 0.34731733798980713, 'tail': 0.3911612927913666, 'thigh_gap': 0.2932726740837097,
            'thighs': 0.4544200003147125, 'very_long_hair': 0.44711941480636597, 'ass': 0.2854885458946228,
            'outdoors': 0.6344638466835022, 'red_eyes': 0.611354410648346, 'day': 0.564970850944519,
            'hair_between_eyes': 0.4444340467453003, 'holding': 0.35846662521362305, 'parted_lips': 0.3867686092853546,
            'blue_sky': 0.3723931908607483, 'cloud': 0.31086698174476624, 'short_sleeves': 0.43279752135276794,
            'sky': 0.3896197974681854, 'gloves': 0.6638736724853516, 'grey_hair': 0.5094802975654602,
            'sweat': 0.4867050349712372, 'navel': 0.6593714952468872, 'crop_top': 0.5243107676506042,
            'shorts': 0.4374789893627167, 'artist_name': 0.3754707872867584, 'midriff': 0.6238733530044556,
            'ass_visible_through_thighs': 0.31088054180145264, 'gym_uniform': 0.37657681107521057,
            'black_shirt': 0.3012588620185852, 'watermark': 0.5147127509117126, 'web_address': 0.6296812295913696,
            'short_shorts': 0.29214906692504883, 'black_shorts': 0.37801358103752136, 'buruma': 0.536261260509491,
            'bike_shorts': 0.35828399658203125, 'black_gloves': 0.4156728982925415, 'sportswear': 0.44427722692489624,
            'baseball_bat': 0.2838006019592285, 'crop_top_overhang': 0.49192047119140625,
            'stomach': 0.36012423038482666, 'black_buruma': 0.3422132134437561,
            'official_alternate_costume': 0.2783987522125244, 'baseball': 0.38377970457077026,
            'baseball_mitt': 0.32592540979385376, 'cropped_shirt': 0.35402947664260864,
            'holding_baseball_bat': 0.2758416533470154, 'black_sports_bra': 0.3463800549507141,
            'sports_bra': 0.28466159105300903, 'exercising': 0.2603980302810669, 'bike_jersey': 0.2661605477333069,
            'patreon_username': 0.7087235450744629, 'patreon_logo': 0.560276210308075
        }, abs=2e-2)
        assert retval[1][2] == pytest.approx({'skadi_(arknights)': 0.5921452641487122}, abs=2e-2)

        assert retval[2][0] == pytest.approx({
            'general': 0.41121846437454224,
            'sensitive': 0.4002530574798584,
            'questionable': 0.03438958525657654,
            'explicit': 0.04617959260940552
        }, abs=2e-2)
        assert retval[2][1] == pytest.approx({
            '1girl': 0.8312125205993652, 'blush': 0.3996567726135254, 'cowboy_shot': 0.28660568594932556,
            'long_hair': 0.7184156775474548, 'long_sleeves': 0.4706878066062927,
            'looking_at_viewer': 0.5503140687942505, 'school_uniform': 0.365602970123291, 'shirt': 0.41183334589004517,
            'sidelocks': 0.28638553619384766, 'smile': 0.3707748055458069, 'solo': 0.520854115486145,
            'standing': 0.2960333526134491, 'tongue': 0.6556028127670288, 'tongue_out': 0.6966925859451294,
            'very_long_hair': 0.5526134371757507, 'skirt': 0.6872812509536743, 'brown_hair': 0.5945607423782349,
            'hair_ornament': 0.4464661478996277, 'hair_ribbon': 0.3646523952484131, 'outdoors': 0.37938451766967773,
            'red_eyes': 0.5426545143127441, 'ribbon': 0.3027467727661133, 'bag': 0.8986430168151855,
            'hair_between_eyes': 0.337802529335022, 'holding': 0.38589367270469666, 'pleated_skirt': 0.6475872993469238,
            'school_bag': 0.666648805141449, 'ahoge': 0.4749193489551544, 'white_shirt': 0.27104783058166504,
            'closed_mouth': 0.28101325035095215, 'collared_shirt': 0.37030768394470215,
            'miniskirt': 0.32576680183410645, ':p': 0.4337637424468994, 'alternate_costume': 0.42441293597221375,
            'black_skirt': 0.34694597125053406, 'twintails': 0.5711237192153931, 'open_clothes': 0.31017544865608215,
            'nail_polish': 0.534726083278656, 'jacket': 0.4544385075569153, 'open_jacket': 0.27831193804740906,
            'flower': 0.45064714550971985, 'plaid_clothes': 0.5494365096092224, 'plaid_skirt': 0.610480546951294,
            'red_flower': 0.35928308963775635, 'contemporary': 0.37732189893722534, 'backpack': 0.5575172305107117,
            'fingernails': 0.27776333689689636, 'cardigan': 0.3264558017253876, 'blue_jacket': 0.31882336735725403,
            'ghost': 0.5534622073173523, 'red_nails': 0.38771501183509827, ':q': 0.3758758008480072,
            'hair_flower': 0.39574217796325684, 'charm_(object)': 0.5394986271858215, 'handbag': 0.37014907598495483,
            'black_bag': 0.44918346405029297, 'shoulder_bag': 0.5881174802780151,
            'symbol-shaped_pupils': 0.5163478255271912, 'blue_cardigan': 0.28089386224746704,
            'black_nails': 0.42480990290641785, 'bag_charm': 0.5010414123535156, 'plum_blossoms': 0.27618563175201416,
            'flower-shaped_pupils': 0.5317837595939636
        }, abs=2e-2)
        assert retval[2][2] == pytest.approx({
            'hu_tao_(genshin_impact)': 0.8859397172927856,
            'boo_tao_(genshin_impact)': 0.7348971366882324
        }, abs=2e-2)
