import pytest

from imgutils.generic.classify_timm import _open_models_for_repo_id, classify_timm_predict
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestGenericClassifyTIMM:
    def test_classify_timm_predict_1(self):
        assert classify_timm_predict(
            get_testfile('classify_timm', 'img1.jpg'),
            repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ) == pytest.approx({
            'jia_redian_ruzi_ruzi': 0.9890832304954529, 'siya_ho': 0.005189628805965185,
            'bai_qi-qsr': 0.0015026535838842392, 'kkuem': 0.0012714712647721171,
            'teddy_(khanshin)': 0.00035598213435150683
        }, abs=1e-2)

    def test_classify_timm_predict_2(self):
        assert classify_timm_predict(
            get_testfile('classify_timm', 'img2.jpg'),
            repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ) == pytest.approx({
            'monori_rogue': 0.6921895742416382, 'stanley_lau': 0.2040979117155075, 'neoartcore': 0.03475344926118851,
            'ayya_sap': 0.005350438412278891, 'goomrrat': 0.004616163671016693
        }, abs=1e-2)

    def test_classify_timm_predict_3(self):
        assert classify_timm_predict(
            get_testfile('classify_timm', 'img3.jpg'),
            repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ) == pytest.approx({
            'shexyo': 0.9998241066932678, 'oroborus': 0.0001537767384434119, 'jeneral': 7.268482477229554e-06,
            'free_style_(yohan1754)': 3.4537688406999223e-06, 'kakeku': 2.5340586944366805e-06
        }, abs=1e-2)

    def test_classify_timm_predict_4(self):
        assert classify_timm_predict(
            get_testfile('classify_timm', 'img4.jpg'),
            repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ) == pytest.approx({
            'z.taiga': 0.9999995231628418, 'tina_(tinafya)': 1.2290533391023928e-07,
            'arind_yudha': 6.17258208990279e-08, 'chixiao': 4.949555076905199e-08,
            'zerotwenty_(020)': 4.218352955831506e-08
        }, abs=1e-2)

    def test_classify_timm_predict_5(self):
        assert classify_timm_predict(
            get_testfile('classify_timm', 'img5.jpg'),
            repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ) == pytest.approx({
            'spam_(spamham4506)': 0.9999998807907104, 'falken_(yutozin)': 4.501828954062148e-08,
            'yuki_(asayuki101)': 3.285677863118508e-08, 'danbal': 5.452678752959628e-09,
            'buri_(retty9349)': 3.757136379789472e-09
        }, abs=1e-2)

    def test_classify_timm_predict_6(self):
        assert classify_timm_predict(
            get_testfile('classify_timm', 'img6.jpg'),
            repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ) == pytest.approx({
            'mashuu_(neko_no_oyashiro)': 1.0, 'minaba_hideo': 4.543745646401476e-08, 'simosi': 6.499865978781827e-09,
            'maoh_yueer': 4.302619149854081e-09, '7nite': 3.6548184478846224e-09
        }, abs=1e-2)

    def test_classify_timm_predict_1_val(self):
        assert classify_timm_predict(
            get_testfile('classify_timm', 'img1.jpg'),
            repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls',
            preprocessor='val'
        ) == pytest.approx({
            'jia_redian_ruzi_ruzi': 0.9890832304954529, 'siya_ho': 0.005189628805965185,
            'bai_qi-qsr': 0.0015026535838842392, 'kkuem': 0.0012714712647721171,
            'teddy_(khanshin)': 0.00035598213435150683
        }, abs=1e-2)

    def test_classify_timm_predict_1_top_score(self):
        top5, all_, top1 = classify_timm_predict(
            get_testfile('classify_timm', 'img1.jpg'),
            repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls',
            fmt=('scores-top5', 'scores', 'scores-top1'),
        )
        assert top5 == pytest.approx({
            'jia_redian_ruzi_ruzi': 0.9890832304954529, 'siya_ho': 0.005189628805965185,
            'bai_qi-qsr': 0.0015026535838842392, 'kkuem': 0.0012714712647721171,
            'teddy_(khanshin)': 0.00035598213435150683
        }, abs=1e-2)
        assert top1 == pytest.approx({
            'jia_redian_ruzi_ruzi': 0.9890832304954529,
        }, abs=1e-2)

        assert len(all_) == 9453
        assert all_['jia_redian_ruzi_ruzi'] == pytest.approx(0.9890832304954529, abs=1e-3)

    def test_classify_timm_predict_1_nps(self):
        embedding, logits, prediction = classify_timm_predict(
            get_testfile('classify_timm', 'img1.jpg'),
            repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls',
            fmt=('embedding', 'logits', 'prediction'),
        )
        assert embedding.shape == (1024,)
        assert logits.shape == (9453,)
        assert prediction.shape == (9453,)
