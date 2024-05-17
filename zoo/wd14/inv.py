import json
import os.path
from typing import Optional

import numpy as np
import onnx
import pandas as pd
from ditk import logging
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from numpy.linalg import pinv
from onnx import AttributeProto, numpy_helper
from tqdm import tqdm

from imgutils.tagging.wd14 import MODEL_NAMES

logging.try_init_root(logging.INFO)


def _make_inverse(model_name, dst_dir: str, onnx_model_file: Optional[str] = None,
                  scale: int = 200, rounds: int = 4, left: float = 5.0, right: float = 12):
    os.makedirs(dst_dir, exist_ok=True)
    model = onnx.load(onnx_model_file or hf_hub_download(
        repo_id='deepghs/wd14_tagger_with_embeddings',
        filename=f'{MODEL_NAMES[model_name]}/model.onnx',
    ))
    names = {}
    for t in model.graph.initializer:
        if any(xn in t.name for xn in ['predictions_dense/MatMul', 'head.fc.weight', 'head.weight']):
            names[t.name] = 'weights'
        if any(xn in t.name for xn in ['predictions_dense/BiasAdd', 'head.fc.bias', 'head.bias']):
            names[t.name] = 'bias'

    node_t = None
    for node in model.graph.node:
        if any(xn in node.name for xn in ['predictions_dense/MatMul_Gemm', 'head/fc/Gemm', 'head/Gemm']):
            assert node_t is None
            node_t = {}
            for item in node.attribute:
                if item.type == AttributeProto.INT:
                    node_t[item.name] = item.i
                elif item.type == AttributeProto.FLOAT:
                    node_t[item.name] = item.f
                elif item.type == AttributeProto.STRING:
                    node_t[item.name] = item.s

    ws = {}
    for t in model.graph.initializer:
        if t.name in names:
            ws[names[t.name]] = t

    ws = {key: numpy_helper.to_array(value) for key, value in ws.items()}

    bias = ws['bias']
    weights = ws['weights']
    if node_t['transB']:
        weights = weights.T

    inv_weights = pinv(weights)
    assert not np.isnan(inv_weights).any()
    assert not np.isinf(inv_weights).any()

    def inv_sigmoid(x):
        return np.log(x) - np.log(1 - x)

    def is_inv_safe(v_epi):
        eps = 10 ** -v_epi
        p = np.concatenate([
            np.ones(10).astype(np.float32),
            np.zeros(10).astype(np.float32),
        ])
        x = np.clip(p, a_min=eps, a_max=1.0 - eps)
        y = inv_sigmoid(x)
        return not bool(np.isnan(y).any() or np.isinf(y).any())

    def get_max_safe_epi(tol=1e-6):
        sl, sr = 1.0, 30.0
        while sl < sr - tol:
            sm = (sl + sr) / 2
            if is_inv_safe(sm):
                sl = sm
            else:
                sr = sm

        return sl

    origin = np.load(hf_hub_download(
        repo_id='deepghs/wd14_tagger_inversion',
        repo_type='dataset',
        filename=f'{model_name}/samples_{scale}.npz'
    ))
    predictions = origin['preds']
    embeddings = origin['embs']

    max_safe_epi = get_max_safe_epi()
    right = min(right, max_safe_epi)
    records = []
    for r in range(rounds):
        xs, ys = [], []
        for epi in tqdm(np.linspace(left, right, 100)):
            pinput = predictions
            eps = 10 ** -epi
            pinput = np.clip(pinput, a_min=eps, a_max=1 - eps)

            if np.isnan(inv_sigmoid(pinput)).any():
                continue
            if np.isinf(inv_sigmoid(pinput)).any():
                continue

            inv_emb = (inv_sigmoid(pinput) - bias) @ inv_weights

            inv_embs = inv_emb / np.linalg.norm(inv_emb, axis=-1)[..., None]
            expected_embs = embeddings / np.linalg.norm(embeddings, axis=-1)[..., None]

            sims = (inv_embs * expected_embs).sum(axis=-1)
            sim = sims.mean()
            records.append({
                'exp_id': len(records),
                'epi': epi,
                'cos_sim': sim,
                **{f'sim_{si}': sv for si, sv in enumerate(sims)},
            })

            xs.append(epi)
            ys.append(sim)
            logging.info(f'Epi: {epi}, cos_sim: {sim}')

        xs = np.array(xs)
        ys = np.array(ys)
        idx = np.argmax(ys)
        logging.info(f'Round #{r}, Best epi: {xs[idx]:.5f}, best sim: {ys[idx]:.6f}')

        rg = right - left
        left, right = xs[idx] - rg * 0.1, xs[idx] + rg * 0.1
        right = min(right, max_safe_epi)

    df = pd.DataFrame(records)
    df = df.sort_values(by=['epi'], ascending=[True])
    df.to_csv(os.path.join(dst_dir, 'inv_experiments.csv'), index=False)

    logging.info(f'Experiment result:\n{df}')
    all_exp_ids = np.array(df['exp_id'])
    idx = np.argmax(df['cos_sim'])
    best_record = df[df['exp_id'] == all_exp_ids[idx]].to_dict('records')[0]
    with open(os.path.join(dst_dir, 'inv_best.json'), 'w') as f:
        json.dump({
            'model': model_name,
            'repository': MODEL_NAMES[model_name],
            'exp_id': best_record['exp_id'],
            'epi': best_record['epi'],
            'cos_sim': best_record['cos_sim'],
            'sims': [best_record[f'sim_{ix}'] for ix in range(scale)],
        }, f, ensure_ascii=False, indent=4, sort_keys=True)

    np.savez(
        os.path.join(dst_dir, 'inv'),
        best_epi=best_record["epi"],
        best_eps=10 ** -best_record["epi"],
        bias=bias,
        inv_weights=inv_weights,
        weights=weights,
    )

    plt.cla()
    logging.info(f'Best epi: {best_record["epi"]:.5f}, best sim: {best_record["cos_sim"]:.6f}')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    axes[0].plot(df["epi"], df["cos_sim"])
    axes[0].set_title(
        f'EPI - Cosine Similarity\nBest epi: {best_record["epi"]:.5f}, best sim: {best_record["cos_sim"]:.6f}')
    axes[0].set_xlabel('EPI')
    axes[0].set_ylabel('Cosine Similarity')

    instances = [value for key, value in best_record.items() if key.startswith('sim_')]
    pd.Series(instances).hist(ax=axes[1], bins=15)
    axes[1].set_title(f'Cosine Similarity Distribution on Best EPI')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Samples')

    plt.savefig(os.path.join(dst_dir, 'inv_plot.png'), dpi=200)
    plt.cla()
