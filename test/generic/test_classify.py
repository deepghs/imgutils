from unittest import skipUnless
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from hbutils.testing import vpip
from huggingface_hub.utils import reset_sessions

from imgutils.generic import classify_predict_score
from imgutils.generic.classify import _open_models_for_repo_id, classify_predict_fmt
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.fixture()
def clean_session():
    reset_sessions()
    _open_models_for_repo_id.cache_clear()
    try:
        yield
    finally:
        reset_sessions()
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestGenericClassify:
    def test_classify_predict_score(self):
        image = Image.open(get_testfile('png_640.png'))
        scores = classify_predict_score(
            image,
            repo_id='deepghs/timms_mobilenet',
            model_name='mobilenetv4_hybrid_medium.ix_e550_r384_in1k',
        )
        assert scores == pytest.approx({
            'n02966687': 0.48493319749832153,
            'n03481172': 0.1228410005569458,
            'n04482393': 0.07170269638299942,
            'n04154565': 0.029927952215075493,
            'n03000684': 0.02070867270231247,
            'n03498962': 0.019339734688401222,
            'n03444034': 0.013918918557465076,
            'n03995372': 0.009074677713215351,
            'n03794056': 0.00785701535642147,
            'n03384352': 0.007194260135293007,
            'n02930766': 0.005858728662133217,
            'n02835271': 0.005415018182247877,
            'n04336792': 0.005253748968243599,
            'n04509417': 0.004338286351412535,
            'n03792782': 0.004189436789602041,
            'n03532672': 0.004000757820904255,
            'n03109150': 0.0034237923100590706,
            'n04517823': 0.0027278559282422066,
            'n03126707': 0.0026790976990014315,
            'n02879718': 0.0026228304486721754
        }, abs=1e-3)

    def test_classify_predict_score_group1(self):
        image = Image.open(get_testfile('png_640.png'))
        scores = classify_predict_score(
            image,
            repo_id='deepghs/timms_mobilenet',
            model_name='mobilenetv4_hybrid_medium.ix_e550_r384_in1k',
            label_group='descriptions',
        )
        assert scores == pytest.approx({
            "carpenter's kit, tool kit": 0.48493319749832153,
            'hammer': 0.1228410005569458,
            'tricycle, trike, velocipede': 0.07170269638299942,
            'screwdriver': 0.029927952215075493,
            'chain saw, chainsaw': 0.02070867270231247,
            'hatchet': 0.019339734688401222,
            'go-kart': 0.013918918557465076,
            'power drill': 0.009074677713215351, 'mousetrap': 0.00785701535642147,
            'forklift': 0.007194260135293007,
            'cab, hack, taxi, taxicab': 0.005858728662133217,
            'bicycle-built-for-two, tandem bicycle, tandem': 0.005415018182247877,
            'stretcher': 0.005253748968243599,
            'unicycle, monocycle': 0.004338286351412535,
            'mountain bike, all-terrain bike, off-roader': 0.004189436789602041,
            'hook, claw': 0.004000757820904255,
            'corkscrew, bottle screw': 0.0034237923100590706,
            'vacuum, vacuum cleaner': 0.0027278559282422066,
            'crane': 0.0026790976990014315,
            'bow': 0.0026228304486721754
        }, abs=1e-3)

    def test_classify_predict_score_group2(self):
        image = Image.open(get_testfile('png_640.png'))
        scores = classify_predict_score(
            image,
            repo_id='deepghs/timms_mobilenet',
            model_name='mobilenetv4_hybrid_medium.ix_e550_r384_in1k',
            label_group='definitions',
        )
        assert scores == pytest.approx({
            "a set of carpenter's tools": 0.48493319749832153,
            'a hand tool with a heavy rigid head and a handle; used to deliver an impulsive force by striking': 0.1228410005569458,
            'a vehicle with three wheels that is moved by foot pedals': 0.07170269638299942,
            'a hand tool for driving screws; has a tip that fits into the head of a screw': 0.029927952215075493,
            'portable power saw; teeth linked to form an endless chain': 0.02070867270231247,
            'a small ax with a short handle used with one hand (usually to chop wood)': 0.019339734688401222,
            'a small low motor vehicle with four wheels and an open framework; used for racing': 0.013918918557465076,
            'a power tool for drilling holes into hard materials': 0.009074677713215351,
            'a trap for catching mice': 0.00785701535642147,
            'a small industrial vehicle with a power operated forked platform in front that can be inserted under loads to lift and move them': 0.007194260135293007,
            'a car driven by a person whose job is to take passengers where they want to go in exchange for money': 0.005858728662133217,
            'a bicycle with two sets of pedals and two seats': 0.005415018182247877,
            'a litter for transporting people who are ill or wounded or dead; usually consists of a sheet of canvas stretched between two poles': 0.005253748968243599,
            'a vehicle with a single wheel that is driven by pedals': 0.004338286351412535,
            'a bicycle with a sturdy frame and fat tires; originally designed for riding in mountainous country': 0.004189436789602041,
            'a mechanical device that is curved or bent to suspend or hold or pull something': 0.004000757820904255,
            'a bottle opener that pulls corks': 0.0034237923100590706,
            'an electrical home appliance that cleans by suction': 0.0027278559282422066,
            'lifts and moves heavy objects; lifting tackle is suspended from a pivoted boom that rotates around a vertical axis': 0.0026790976990014315,
            'a weapon for shooting arrows, composed of a curved piece of resilient wood with a taut cord to propel the arrow': 0.0026228304486721754
        }, abs=1e-3)

    def test_classify_predict_score_top5(self):
        image = Image.open(get_testfile('png_640.png'))
        scores = classify_predict_score(
            image,
            repo_id='deepghs/timms_mobilenet',
            model_name='mobilenetv4_hybrid_medium.ix_e550_r384_in1k',
            topk=5,
        )
        assert scores == pytest.approx({
            'n02966687': 0.48493319749832153,
            'n03481172': 0.1228410005569458,
            'n04482393': 0.07170269638299942,
            'n04154565': 0.029927952215075493,
            'n03000684': 0.02070867270231247,
        }, abs=1e-3)

    def test_classify_predict_score_top5_group1(self):
        image = Image.open(get_testfile('png_640.png'))
        scores = classify_predict_score(
            image,
            repo_id='deepghs/timms_mobilenet',
            model_name='mobilenetv4_hybrid_medium.ix_e550_r384_in1k',
            label_group='descriptions',
            topk=5,
        )
        assert scores == pytest.approx({
            "carpenter's kit, tool kit": 0.48493319749832153,
            'hammer': 0.1228410005569458,
            'tricycle, trike, velocipede': 0.07170269638299942,
            'screwdriver': 0.029927952215075493,
            'chain saw, chainsaw': 0.02070867270231247,
        }, abs=1e-3)

    def test_classify_predict_fmt(self):
        image = Image.open(get_testfile('png_640.png'))
        results = classify_predict_fmt(
            image,
            repo_id='deepghs/timms_mobilenet',
            model_name='mobilenetv4_hybrid_medium.ix_e550_r384_in1k',
        )
        assert results == pytest.approx({
            'n02966687': 0.48493319749832153,
            'n03481172': 0.1228410005569458,
            'n04482393': 0.07170269638299942,
            'n04154565': 0.029927952215075493,
            'n03000684': 0.02070867270231247
        }, abs=1e-3)

    def test_classify_predict_fmt_complex(self):
        image = Image.open(get_testfile('png_640.png'))
        results = classify_predict_fmt(
            image,
            repo_id='deepghs/timms_mobilenet',
            model_name='mobilenetv4_hybrid_medium.ix_e550_r384_in1k',
            fmt={
                'scores-top10': 'scores-top10',
                'scores-top10-descriptions': 'scores-top10-descriptions',
                'scores-top5-definitions': 'scores-top5-definitions',
                'scores-top5-descriptions': 'scores-top5-descriptions',
                'embedding': 'embedding',
            }
        )
        assert results['scores-top10'] == pytest.approx({
            'n02966687': 0.48493319749832153,
            'n03481172': 0.1228410005569458,
            'n04482393': 0.07170269638299942,
            'n04154565': 0.029927952215075493,
            'n03000684': 0.02070867270231247,
            'n03498962': 0.019339734688401222,
            'n03444034': 0.013918918557465076,
            'n03995372': 0.009074677713215351,
            'n03794056': 0.00785701535642147,
            'n03384352': 0.007194260135293007
        }, abs=1e-3)
        assert results['scores-top10-descriptions'] == pytest.approx({
            "carpenter's kit, tool kit": 0.48493319749832153,
            'hammer': 0.1228410005569458,
            'tricycle, trike, velocipede': 0.07170269638299942,
            'screwdriver': 0.029927952215075493,
            'chain saw, chainsaw': 0.02070867270231247,
            'hatchet': 0.019339734688401222,
            'go-kart': 0.013918918557465076,
            'power drill': 0.009074677713215351,
            'mousetrap': 0.00785701535642147,
            'forklift': 0.007194260135293007
        }, abs=1e-3)
        assert results['scores-top5-definitions'] == pytest.approx({
            "a set of carpenter's tools": 0.48493319749832153,
            'a hand tool with a heavy rigid head and a handle; used to deliver an impulsive force by striking': 0.1228410005569458,
            'a vehicle with three wheels that is moved by foot pedals': 0.07170269638299942,
            'a hand tool for driving screws; has a tip that fits into the head of a screw': 0.029927952215075493,
            'portable power saw; teeth linked to form an endless chain': 0.02070867270231247
        }, abs=1e-3)
        assert results['scores-top5-descriptions'] == pytest.approx({
            "carpenter's kit, tool kit": 0.48493319749832153,
            'hammer': 0.1228410005569458,
            'tricycle, trike, velocipede': 0.07170269638299942,
            'screwdriver': 0.029927952215075493,
            'chain saw, chainsaw': 0.02070867270231247
        }, abs=1e-3)
        # np.save(get_testfile('png_640_emb.npy'), results['embedding'])
        assert results['embedding'].shape == (1280,)
        expected_embedding = np.load(get_testfile('png_640_emb.npy'))
        emb_1 = results['embedding'] / np.linalg.norm(results['embedding'], axis=-1, keepdims=True)
        emb_2 = expected_embedding / np.linalg.norm(expected_embedding, axis=-1, keepdims=True)
        emb_sims = (emb_1 * emb_2).sum()
        assert emb_sims >= 0.99, 'Direction not match with expected embedding.'
        assert np.linalg.norm(results['embedding']) == pytest.approx(np.linalg.norm(expected_embedding))

    @patch("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    @skipUnless(vpip('huggingface_hub') < '0.34', 'Has problem on huggingface 0.34+')
    def test_classify_predict_score_top5_offline_mode(self, clean_session):
        image = Image.open(get_testfile('png_640.png'))
        scores = classify_predict_score(
            image,
            repo_id='deepghs/timms_mobilenet',
            model_name='mobilenetv4_hybrid_medium.ix_e550_r384_in1k',
            topk=5,
        )
        assert scores == pytest.approx({
            'n02966687': 0.48493319749832153,
            'n03481172': 0.1228410005569458,
            'n04482393': 0.07170269638299942,
            'n04154565': 0.029927952215075493,
            'n03000684': 0.02070867270231247,
        }, abs=1e-3)
