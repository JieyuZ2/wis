from abc import ABC, abstractmethod
from collections import Counter
from functools import partial
from itertools import chain
from typing import List, Union, Callable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import check_random_state

ABSTAIN = -1


class Expression(ABC):
    @abstractmethod
    def apply(self, x: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def include(self, other):
        raise NotImplementedError

    @abstractmethod
    def exclude(self, other):
        raise NotImplementedError

    def overlap(self, other):
        if self.exclude(other): return False
        if self.include(other): return False
        if other.include(self): return False
        return True


class UnaryExpression(Expression):
    def __init__(self, idx, threshold):
        self.idx = idx
        self.threshold = threshold

    def apply(self, x: np.ndarray):
        assert x.ndim == 2, 'dimension of x should be 2!'
        return self.apply_(x[:, self.idx])

    @abstractmethod
    def apply_(self, x: np.ndarray):
        raise NotImplementedError

    def include(self, other: Expression):
        if isinstance(other, UnaryExpression):
            if self.idx == other.idx:
                return self.include_(other)
            return False
        if isinstance(other, BinaryExpression):
            return self.include(other.e1) and self.include(other.e2)

    @abstractmethod
    def include_(self, other: Expression):
        raise NotImplementedError

    def exclude(self, other: Expression):
        if isinstance(other, UnaryExpression):
            if self.idx == other.idx:
                return self.exclude_(other)
            return True
        if isinstance(other, BinaryExpression):
            return self.exclude(other.e1) and self.exclude(other.e2)

    @abstractmethod
    def exclude_(self, other: Expression):
        raise NotImplementedError

    def __str__(self):
        s = f'=====[{self.__class__}]=====\n'
        s += f'[idx] {self.idx}\n'
        s += f'[threshold] {self.threshold}\n'
        return s


class GreaterExpression(UnaryExpression):
    def apply_(self, x: np.ndarray):
        return x > self.threshold

    def include_(self, other: Expression):
        if isinstance(other, GreaterExpression) or isinstance(other, EqualExpression):
            return other.threshold > self.threshold
        if isinstance(other, InIntervalExpression):
            return other.threshold[0] > self.threshold
        return False

    def exclude_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return other.threshold < self.threshold
        if isinstance(other, InIntervalExpression):
            return other.threshold[1] < self.threshold
        if isinstance(other, LessExpression):
            return other.threshold < self.threshold
        return False


class LessExpression(UnaryExpression):
    def apply_(self, x: np.ndarray):
        return x < self.threshold

    def include_(self, other: Expression):
        if isinstance(other, LessExpression) or isinstance(other, EqualExpression):
            return other.threshold < self.threshold
        if isinstance(other, InIntervalExpression):
            return other.threshold[1] < self.threshold
        return False

    def exclude_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return other.threshold > self.threshold
        if isinstance(other, InIntervalExpression):
            return other.threshold[0] > self.threshold
        if isinstance(other, GreaterExpression):
            return other.threshold > self.threshold
        return False


class EqualExpression(UnaryExpression):
    def apply_(self, x: np.ndarray):
        return x == self.threshold

    def include_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return other.threshold == self.threshold
        return False

    def exclude_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return other.threshold != self.threshold
        else:
            return other.exclude(self)


class InIntervalExpression(UnaryExpression):
    def apply_(self, x: np.ndarray):
        return np.logical_and(self.threshold[0] < x, x < self.threshold[1])

    def include_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return self.threshold[0] < other.threshold < self.threshold[1]
        if isinstance(other, InIntervalExpression):
            return self.threshold[0] < other.threshold[0] and other.threshold[1] < self.threshold[1]
        return False

    def exclude_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return other.threshold < self.threshold[0] or other.threshold > self.threshold[1]
        if isinstance(other, InIntervalExpression):
            return other.threshold[0] > self.threshold[1] or other.threshold[1] < self.threshold[0]
        return other.exclude(self)


class OutIntervalExpression(UnaryExpression):
    def apply_(self, x: np.ndarray):
        return np.logical_or(self.threshold[0] > x, x > self.threshold[1])

    def include_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return self.threshold[0] > other.threshold or other.threshold > self.threshold[1]
        if isinstance(other, GreaterExpression):
            return self.threshold[1] < other.threshold
        if isinstance(other, LessExpression):
            return self.threshold[0] > other.threshold
        if isinstance(other, InIntervalExpression):
            return self.threshold[0] > other.threshold[1] or other.threshold[0] > self.threshold[1]
        if isinstance(other, OutIntervalExpression):
            return self.threshold[0] > other.threshold[0] and other.threshold[1] > self.threshold[1]
        return False

    def exclude_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return self.threshold[0] < other.threshold < self.threshold[1]
        if isinstance(other, InIntervalExpression):
            return other.threshold[0] > self.threshold[0] and other.threshold[1] < self.threshold[1]
        return False


class BinaryExpression(Expression):
    logic_op: Callable

    def __init__(self, e1: Expression, e2: Expression):
        self.e1 = e1
        self.e2 = e2

    def apply(self, x: np.ndarray):
        x1 = self.e1.apply(x)
        x2 = self.e2.apply(x)
        return self.logic_op(x1, x2)

    def include(self, other: Expression):
        if isinstance(other, UnaryExpression):
            return self.e1.include(other) or self.e2.include(other)
        if isinstance(other, BinaryExpression):
            e1_included = self.e1.include(other.e1) or self.e2.include(other.e1)
            e2_included = self.e1.include(other.e2) or self.e2.include(other.e2)
            return e1_included and e2_included

    def exclude(self, other: Expression):
        if isinstance(other, UnaryExpression):
            return self.e1.exclude(other) and self.e2.exclude(other)
        if isinstance(other, BinaryExpression):
            e1_excluded = self.e1.exclude(other.e1) and self.e2.exclude(other.e1)
            e2_excluded = self.e1.exclude(other.e2) and self.e2.exclude(other.e2)
            return e1_excluded and e2_excluded


class AndExpression(BinaryExpression):
    logic_op = staticmethod(np.logical_and)


class OrExpression(BinaryExpression):
    logic_op = staticmethod(np.logical_or)


class NGramExpression(Expression):
    def __init__(self, idx, threshold, ngram):
        self.idx = idx
        self.threshold = threshold
        self.ngram = ngram

    def apply(self, x: np.ndarray):
        assert x.ndim == 2, 'dimension of x should be 2!'
        applied = x[:, self.idx] > self.threshold
        if isinstance(applied, csr_matrix):
            applied = applied.toarray().squeeze()
        return applied

    def include(self, other):
        raise NotImplementedError

    def exclude(self, other):
        raise NotImplementedError

    def __str__(self):
        s = f'=====[{self.__class__}]=====\n'
        s += f'[idx] {self.idx}\n'
        s += f'[threshold] {self.threshold}\n'
        s += f'[ngram] {self.ngram}\n'
        return s


class LF:
    def __init__(self, e: Expression, label: int, acc: float = -1.0, propensity: float = -1.0):
        self.e = e
        self.label = label
        self.acc = acc
        self.propensity = propensity

    def apply(self, x: np.ndarray):
        x = self.e.apply(x)
        return x * self.label + (1 - x) * ABSTAIN


class AbstractLFApplier:
    def __init__(self, lf_list: List[LF]):
        self.lfs = lf_list
        self.labels = [r.label for r in lf_list]
        self.accs = [r.acc for r in lf_list]

    @abstractmethod
    def apply(self, dataset):
        raise NotImplementedError

    def __len__(self):
        return len(self.lfs)


class NGramLFApplier(AbstractLFApplier):
    def __init__(self, lf_list: List[LF], vectorizer: CountVectorizer):
        super().__init__(lf_list)
        self.vectorizer = vectorizer

    def apply(self, corpus):
        X = self.vectorizer.transform(corpus)
        L = np.stack([lf.apply(X) for lf in self.lfs]).T
        return L


class NoEnoughLFError(Exception):
    def __init__(self, label=None):
        if label is None:
            self.message = 'cannot find enough lfs, please lower the min support or the min acc gain!'
        else:
            self.message = f'cannot find any lf for label {label}, please lower the min support or the min acc gain!'
        super().__init__(self.message)


class AbstractLFGenerator(ABC):
    lf_applier_type: Callable
    X: Union[np.ndarray, csr_matrix]
    label_to_candidate_lfs: dict

    def __init__(self,
                 dataset: List[str],
                 y: np.ndarray,
                 target_labels: List[int] = None,
                 min_acc_gain: float = 0.1,
                 min_support: float = 0.01,
                 random_state=None
                 ):
        self.Y = y
        self.n_class = len(set(self.Y))
        self.target_labels = target_labels if target_labels is not None else list(range(self.n_class))
        assert self.n_class > 1
        self.dataset = dataset
        self.n_data = len(dataset)
        self.min_support = int(min_support * self.n_data)
        self.min_acc_gain = min_acc_gain
        self.class_marginal = self.array_to_marginals(self.Y)

        self.generator = check_random_state(random_state)

    @staticmethod
    def array_to_marginals(y):
        class_counts = Counter(y)
        sorted_counts = np.array([v for k, v in sorted(class_counts.items())])
        _marginal = sorted_counts / sum(sorted_counts)
        return _marginal

    @staticmethod
    def calc_acc(y):
        return np.sum(y) / len(y)

    def check_candidate_lfs_enough_(self, n_lfs: Union[int, List[int]]):
        if isinstance(n_lfs, int):
            assert sum(map(len, self.label_to_candidate_lfs.values())) > n_lfs, NoEnoughLFError()
        else:
            assert len(n_lfs) == len(self.target_labels)
            for label, n_lfs_i in zip(self.target_labels, n_lfs):
                assert len(self.label_to_candidate_lfs[label]) > n_lfs_i, NoEnoughLFError(label)

    def return_candidate_lfs(self):
        return list(chain.from_iterable(self.label_to_candidate_lfs.values()))

    def generate(self, mode: str, **kwargs):
        if mode == 'random':
            return self.random_generate(**kwargs)
        if mode == 'accurate':
            return self.accurate_generate(**kwargs)
        raise NotImplementedError(f'generate mode {mode} is not implemented!')

    def random_generate(self, n_lfs: Union[int, List[int]] = 10, duplicated_lf=False) -> AbstractLFApplier:
        if not duplicated_lf:
            self.check_candidate_lfs_enough_(n_lfs)
        if isinstance(n_lfs, int):
            candidate_lfs = self.return_candidate_lfs()
            lfs = list(self.generator.choice(candidate_lfs, n_lfs, replace=duplicated_lf))
        else:
            lfs = []
            for label, n_lfs_i in zip(self.target_labels, n_lfs):
                candidate_lfs = self.label_to_candidate_lfs[label]
                lfs_i = list(self.generator.choice(candidate_lfs, n_lfs_i, replace=duplicated_lf))
                lfs += lfs_i
        return self.lf_applier_type(lfs)

    def accurate_generate(self, n_lfs: Union[int, List[int]] = 10) -> AbstractLFApplier:
        self.check_candidate_lfs_enough_(n_lfs)
        if isinstance(n_lfs, int):
            candidate_lfs = self.return_candidate_lfs()
            lfs = sorted(candidate_lfs, key=lambda x: -x.acc)[:n_lfs]
        else:
            lfs = []
            for label, n_lfs_i in zip(self.target_labels, n_lfs):
                candidate_lfs = self.label_to_candidate_lfs[label]
                lfs += sorted(candidate_lfs, key=lambda x: -x.acc)[:n_lfs_i]
        return self.lf_applier_type(lfs)


class NGramLFGenerator(AbstractLFGenerator):
    def __init__(self,
                 dataset: List[str],
                 y: np.ndarray,
                 target_labels: List[int] = None,
                 vectorizer: CountVectorizer = None,
                 ngram_range=(1, 1),
                 min_acc_gain: float = 0.1,
                 min_support: float = 0.01,
                 random_state=None
                 ):

        super(NGramLFGenerator, self).__init__(dataset, y, target_labels, min_acc_gain, min_support, random_state)
        if vectorizer is None:
            vectorizer = CountVectorizer(strip_accents='ascii',
                                         # stop_words='english',
                                         ngram_range=ngram_range,
                                         analyzer='word',
                                         max_df=0.90,
                                         min_df=self.min_support / self.n_data,
                                         max_features=None,
                                         vocabulary=None,
                                         binary=False)

        self.X = vectorizer.fit_transform(dataset)
        self.vectorizer = vectorizer
        self.idx_to_ngram = vectorizer.get_feature_names()
        self.n_feature = self.X.shape[1]
        self.label_to_candidate_lfs = self.generate_label_to_lfs()
        self.lf_applier_type = partial(NGramLFApplier, vectorizer=vectorizer)

    def generate_label_to_lfs(self):
        label_to_lfs = {}
        for label in self.target_labels:
            y = np.array(self.Y == label, dtype=np.int)
            min_acc = self.class_marginal[label] + self.min_acc_gain
            lfs = []
            for idx in range(self.n_feature):
                x = self.X[:, idx].toarray().squeeze()
                exist_idx = x > 0
                exist_acc = self.calc_acc(y[exist_idx])
                if exist_acc > min_acc and np.sum(exist_idx) > self.min_support:
                    ngram = self.idx_to_ngram[idx]
                    propensity = np.sum(exist_idx) / self.n_data
                    e = NGramExpression(idx=idx, threshold=0, ngram=ngram)
                    lf = LF(e=e, label=label, acc=exist_acc, propensity=propensity)
                    lfs.append(lf)
            assert len(lfs) > 1, f'cannot find any lf for label {label}, please lower the min support or the min acc gain!'
            label_to_lfs[label] = lfs
        return label_to_lfs
