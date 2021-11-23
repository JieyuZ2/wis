from functools import partial

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import PredefinedSplit

from ..dataset import TextDataset


def train_text_multiple(data_train: TextDataset, data_val: TextDataset,
                        y_train: np.ndarray, y_valid: np.ndarray, **kwargs):
    X = data_train.X
    X_val = data_val.X

    y_val_cnt = np.unique(data_val.y, return_counts=True)[1]
    class_prior = y_val_cnt / y_val_cnt.sum()
    # attribute_prior = y_train.sum(0) / len(y_train)

    classifier = []
    m = y_train.shape[1]
    for i in range(m):
        y = y_train[:, i]
        y_val = y_valid[:, i]
        X_ = np.vstack([X, X_val])
        y_ = np.hstack([y, y_val])

        clf = None
        try:
            idx = np.zeros(len(y_), dtype=int)
            idx[:len(y)] = -1
            cv = PredefinedSplit(test_fold=idx)

            f1 = make_scorer(partial(f1_score, average='macro'))

            clf = LogisticRegressionCV(cv=cv, scoring=f1, solver='lbfgs', max_iter=1000, fit_intercept=False).fit(X_, y_)

        except:
            pass

        classifier.append(clf)

    return classifier


def apply_multiple(models, dataset, desired_class_to_attributes):
    n_class = len(desired_class_to_attributes)
    X = dataset.X
    A = np.zeros((len(X), len(models)))
    for i, model in enumerate(models):
        if model is not None:
            A[:, i] = model.predict_proba(X)[:, 1]
        else:
            A[:, i] = 0.5

    proba = np.zeros((len(X), n_class))
    for i, a in enumerate(desired_class_to_attributes):
        Ai = np.prod(A[:, a.astype(bool)], axis=1) * np.prod((1 - A)[:, (1 - a).astype(bool)], axis=1)
        proba[:, i] = Ai

    proba /= proba.sum(1, keepdims=True)
    proba = np.nan_to_num(proba)
    for i, p in enumerate(proba):
        if p.sum() == 0:
            proba[i] = 1 / n_class

    return proba


def train_text(data_train: TextDataset, data_val: TextDataset = None, y: np.ndarray = None, **kwargs):
    X = data_train.X
    if y is None:
        y = data_train.y

    try:

        if data_val is not None:
            X_val = data_val.X
            y_val = data_val.y
            X_ = np.vstack([X, X_val])
            y_ = np.hstack([y, y_val])

            idx = np.zeros(len(y_), dtype=int)
            idx[:len(y)] = -1
            cv = PredefinedSplit(test_fold=idx)

            f1 = make_scorer(partial(f1_score, average='macro'))

            clf = LogisticRegressionCV(cv=cv, scoring=f1, solver='lbfgs', max_iter=1000, fit_intercept=False).fit(X_, y_)
        else:
            clf = LogisticRegression(solver='lbfgs', max_iter=1000, **kwargs).fit(X, y)

    except:
        n_class = len(data_train.classes)
        clf = train_proba_text(data_train, data_val, y=np.eye(n_class)[y])

    return clf


def train_proba_text(data_train: TextDataset, data_val: TextDataset = None, y: np.ndarray = None, **kwargs):
    X = data_train.X
    n, n_class = y.shape
    yy = np.concatenate([np.ones(n) * c for c in range(n_class)])
    xx = np.concatenate([X] * n_class, axis=0)
    sample_weight = y.flatten('F')

    if data_val is not None:
        X_val = data_val.X
        y_val = data_val.y
        X_ = np.vstack([xx, X_val])
        y_ = np.hstack([yy, y_val])
        sample_weight = np.hstack([sample_weight, np.ones(len(y_val))])

        idx = np.zeros(len(y_))
        idx[:len(yy)] = -1
        cv = PredefinedSplit(test_fold=idx)

        f1 = make_scorer(partial(f1_score, average='macro'))

        clf = LogisticRegressionCV(cv=cv, scoring=f1, solver='lbfgs', max_iter=1000, fit_intercept=False).fit(X_, y_, sample_weight=sample_weight)
    else:
        clf = LogisticRegression(solver='lbfgs', max_iter=1000, **kwargs).fit(xx, yy, sample_weight=sample_weight)

    return clf


def test_text(model, data_test: TextDataset, convert_pred=False):
    probas = model.predict_proba(data_test.X)
    preds = model.predict(data_test.X)

    n_class = len(data_test.classes)
    if probas.shape[1] < n_class:
        true_probas = np.zeros((len(preds), n_class))
        true_probas[:, model.classes_] = probas
        probas = true_probas
    if convert_pred:
        preds = np.array(data_test.classes)[preds]
    return probas, preds
