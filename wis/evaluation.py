from functools import partial
from typing import List

import numpy as np
import sklearn.metrics as cls_metric
from snorkel.utils import probs_to_preds


def metric_to_direction(metric: str) -> str:
    if metric in ['acc', 'f1_binary', 'f1_micro', 'f1_macro', 'f1_weighted', 'auc']:
        return 'maximize'
    if metric in ['logloss', 'brier']:
        return 'minimize'
    raise NotImplementedError(f'cannot automatically decide the direction for {metric}!')


def brier_score_loss(golden_labels, prob_labels):
    r = len(np.unique(golden_labels))
    return np.mean((np.eye(r)[golden_labels] - prob_labels) ** 2)


def accuracy_score_(y_true: np.ndarray, y_proba: np.ndarray, **kwargs):
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.accuracy_score(y_true, y_pred)


def f1_score_(y_true: np.ndarray, y_proba: np.ndarray, average: str, **kwargs):
    if average == 'binary' and len(np.unique(y_true)) > 2:
        return 0.0
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.f1_score(y_true, y_pred, average=average)


def lowest_class_f1_(y_true: np.ndarray, y_proba: np.ndarray, **kwargs):
    y_pred = probs_to_preds(y_proba, **kwargs)
    return np.min(cls_metric.f1_score(y_true, y_pred, average=None))


def recall_score_(y_true: np.ndarray, y_proba: np.ndarray, average: str, **kwargs):
    if average == 'binary' and len(np.unique(y_true)) > 2:
        return 0.0
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.recall_score(y_true, y_pred, average=average)


def precision_score_(y_true: np.ndarray, y_proba: np.ndarray, average: str, **kwargs):
    if average == 'binary' and len(np.unique(y_true)) > 2:
        return 0.0
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.precision_score(y_true, y_pred, average=average)


def auc_score_(y_true: np.ndarray, y_proba: np.ndarray, **kwargs):
    if len(np.unique(y_true)) > 2:
        return 0.0
    fpr, tpr, thresholds = cls_metric.roc_curve(y_true, y_proba[:, 1], pos_label=1)
    return cls_metric.auc(fpr, tpr)


METRIC = {
    'acc'               : accuracy_score_,
    'auc'               : auc_score_,
    'lowest_class_f1'   : lowest_class_f1_,
    'f1_binary'         : partial(f1_score_, average='binary'),
    'f1_micro'          : partial(f1_score_, average='micro'),
    'f1_macro'          : partial(f1_score_, average='macro'),
    'f1_weighted'       : partial(f1_score_, average='weighted'),
    'recall_binary'     : partial(recall_score_, average='binary'),
    'recall_micro'      : partial(recall_score_, average='micro'),
    'recall_macro'      : partial(recall_score_, average='macro'),
    'recall_weighted'   : partial(recall_score_, average='weighted'),
    'precision_binary'  : partial(precision_score_, average='binary'),
    'precision_micro'   : partial(precision_score_, average='micro'),
    'precision_macro'   : partial(precision_score_, average='macro'),
    'precision_weighted': partial(precision_score_, average='weighted'),
    'logloss'           : cls_metric.log_loss,
    'brier'             : brier_score_loss,
}


class AverageMeter:
    def __init__(self, names: List[str]):
        self.named_dict = {n: [] for n in names}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.named_dict[k].append(v)

    def get_results(self):
        results = {}
        for n, l in self.named_dict.items():
            if len(l) > 0:
                results[n] = (np.mean(l), np.std(l))
        return results
