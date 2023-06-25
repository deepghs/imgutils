import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from hbutils.random import keep_global_state
from hbutils.system import TemporaryDirectory
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal


def _pos_neg_to_true_score(pos, neg):
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_value = np.concatenate([pos, neg])

    return y_true, y_value


def plt_confusion_matrix(ax, y_true, y_pred, title: str = 'Confusion Matrix',
                         normalize: Literal['true', 'pred', None] = 'true', cmap=None):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Diff', 'Sim'],
    )
    disp.plot(ax=ax, cmap=cmap or plt.cm.Blues)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    ax.set_title(title)


@keep_global_state()
def _create_score_curve(ax, name, func, pos, neg, title=None, units: int = 2000,
                        xrange: Tuple[float, float] = (0.0, 1.0)):
    y_true, y_score = _pos_neg_to_true_score(pos, neg)
    xs, ys = [], []
    scores = np.sort(y_score, kind='heapsort')
    if len(scores) > units:
        scores = np.random.choice(scores, units)
    for score in np.sort(scores, kind='heapsort'):
        _y_pred = y_score >= score
        precision = func(y_true, _y_pred, zero_division=1)
        xs.append(score)
        ys.append(precision)

    xs = np.array(xs)
    ys = np.array(ys)
    maxj = np.argmax(ys)
    ax.plot(xs, ys, label=f'{ys[maxj]:.2f} at {xs[maxj]:.3f}')

    ax.set_xlabel(f'score')
    ax.set_ylabel(f'{name}')
    ax.set_xlim(xrange)
    ax.set_ylim([0.0, 1.0])
    ax.set_title(title or f'{name} curve'.capitalize())
    ax.grid()
    ax.legend()


def plt_f1_curve(ax, pos, neg, title='F1 Curve', units: int = 2000,
                 xrange: Tuple[float, float] = (0.0, 1.0)):
    _create_score_curve(ax, 'F1', f1_score, pos, neg, title, units, xrange)


def plt_p_curve(ax, pos, neg, title='Precision Curve', units: int = 2000,
                xrange: Tuple[float, float] = (0.0, 1.0)):
    _create_score_curve(ax, 'precision', precision_score, pos, neg, title, units, xrange)


def plt_r_curve(ax, pos, neg, title='Recall Curve', units: int = 2000,
                xrange: Tuple[float, float] = (0.0, 1.0)):
    _create_score_curve(ax, 'recall', recall_score, pos, neg, title, units, xrange)


def plt_pr_curve(ax, pos, neg, title='PR Curve'):
    y_true, y_score = _pos_neg_to_true_score(pos, neg)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    _map = -np.trapz(precision, recall)
    disp.plot(ax=ax, name=f'mAP {_map:.3f}')

    ax.set_title(title)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid()
    ax.legend()


def plt_roc_curve(ax, pos, neg, title: str = 'ROC Curve'):
    y_true, y_score = _pos_neg_to_true_score(pos, neg)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_value = auc(fpr, tpr)

    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_value)
    display.plot(ax=ax)

    ax.set_title(title)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid()
    ax.legend()


def get_threshold_with_f1(pos, neg, units: int = 2000):
    y_true, y_score = _pos_neg_to_true_score(pos, neg)
    xs, ys = [], []
    scores = np.sort(y_score, kind='heapsort')
    if len(scores) > units:
        scores = np.random.choice(scores, units)
    for score in np.sort(scores, kind='heapsort'):
        _y_pred = y_score >= score
        precision = f1_score(y_true, _y_pred, zero_division=1)
        xs.append(score)
        ys.append(precision)

    xs = np.array(xs)
    ys = np.array(ys)
    maxj = np.argmax(ys)
    return xs[maxj].item(), ys[maxj].item()


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, dict):
        return type(x)({key: _to_numpy(value) for key, value in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([_to_numpy(item) for item in x])
    else:
        return x


def plt_export(func, *args, figsize=(6, 6), **kwargs) -> Image.Image:
    fig = plt.Figure(figsize=figsize)
    fig.tight_layout()
    func(fig.gca(), *_to_numpy(args), **_to_numpy(kwargs))

    with TemporaryDirectory() as td:
        imgfile = os.path.join(td, 'image.png')
        fig.savefig(imgfile)

        image = Image.open(imgfile)
        image.load()
        image = image.convert('RGB')
        return image
