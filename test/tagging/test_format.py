import pytest

from imgutils.tagging import tags_to_text


@pytest.fixture()
def tag_mapping_full():
    return {
        '1girl': 0.9988248348236084, 'solo': 0.9909533262252808, 'long_hair': 0.9280524849891663,
        'breasts': 0.9894732236862183, 'looking_at_viewer': 0.6717687249183655, 'blush': 0.7718942165374756,
        'smile': 0.8541756868362427, 'large_breasts': 0.8418245315551758, 'shirt': 0.8424559235572815,
        'long_sleeves': 0.48852404952049255, 'hat': 0.9111874103546143, 'holding': 0.5431409478187561,
        'hair_between_eyes': 0.6264550685882568, 'brown_eyes': 0.5628016591072083, 'underwear': 0.9418268203735352,
        'collarbone': 0.4373166263103485, 'panties': 0.958938717842102, 'white_shirt': 0.6890810132026672,
        'grey_hair': 0.586448073387146, 'cowboy_shot': 0.4767565429210663, 'censored': 0.894860029220581,
        'open_clothes': 0.8671413064002991, 'off_shoulder': 0.3682156801223755, 'character_name': 0.4490547776222229,
        'cup': 0.8550489544868469, 'black_panties': 0.8905943632125854, 'open_shirt': 0.6449720859527588,
        'half-closed_eyes': 0.6000269055366516, 'thick_eyebrows': 0.7797143459320068, 'panty_pull': 0.6826801300048828,
        'convenient_censoring': 0.39101117849349976, 'areola_slip': 0.41196826100349426, 'alcohol': 0.6968570351600647,
        'drinking_glass': 0.9340789318084717, 'mini_hat': 0.774779200553894, 'drunk': 0.6200403571128845,
        'wine_glass': 0.9240747690200806, 'hair_over_breasts': 0.35023659467697144,
        'tilted_headwear': 0.6744506359100342, 'wine': 0.5849414467811584, 'novelty_censor': 0.8529061079025269,
        'censored_nipples': 0.3598130941390991
    }


@pytest.fixture()
def tag_mapping():
    return {
        'panty_pull': 0.6826801300048828, 'panties': 0.958938717842102, 'drinking_glass': 0.9340789318084717,
        'areola_slip': 0.41196826100349426, '1girl': 0.9988248348236084
    }


@pytest.mark.unittest
class TestTaggingFormat:
    def test_tags_to_text(self, tag_mapping):
        assert tags_to_text(tag_mapping) == '1girl, panties, drinking_glass, panty_pull, areola_slip'
        assert tags_to_text(tag_mapping, use_spaces=True) == \
               '1girl, panties, drinking glass, panty pull, areola slip'
        assert tags_to_text(tag_mapping, use_spaces=True, include_ranks=True) == \
               '(1girl:0.999), (panties:0.959), (drinking glass:0.934), (panty pull:0.683), (areola slip:0.412)'
