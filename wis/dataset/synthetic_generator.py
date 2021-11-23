from collections import namedtuple
from itertools import chain, permutations
from typing import Tuple

import numpy as np

from .dataset import BaseDataset
from ..label import ABSTAIN, EQUAL, INCLUSION, INCLUDED, EXCLUSIVE, OVERLAP, reverse_relation
from ..mytype import Config
from ..utils import check_random_state, replace_array

Dataset = namedtuple('Dataset', ['X', 'Y', 'Ys', 'D'])


class SyntheticDatasetConfig(Config):
    graph_type: str = 'dag'

    n_data: int = 10000
    n_samples: int = 1
    n_desired_labels: int = 5
    n_labels: int = 10

    propensity_range: Tuple = (1.0, 1.0)
    accuracy_range: Tuple = (0.7, 0.9)
    overlap_factor: float = 0.5
    subsuming_factor: float = 0.2

    n_models_per_label: int = 5

    n_models: int = 20
    model_label_space_lo: int = 2
    model_label_space_hi: int = 3

    seed: int = 12345


class SyntheticDataset(BaseDataset):
    def __init__(self, L, y, classes, model_class, label_relations):
        self.classes = classes
        self.model_class = model_class
        self.label_relations = label_relations
        self.L = L
        self.y = y

        L_local = np.zeros_like(L)
        for i, m in enumerate(model_class):
            L_local[:, i] = np.where(L[:, i] == ABSTAIN, ABSTAIN, replace_array(L[:, i], dict(zip(m, range(len(m))))))
        self.L_local = L_local

    def get_subset(self, idx):
        return SyntheticDataset(
            L=self.L[idx],
            y=self.y[idx],
            classes=self.classes,
            model_class=self.model_class,
            label_relations=self.label_relations
        )


class SyntheticGenerator(object):
    def __init__(
            self,
            desired_classes,
            model_classes,
            label_relations,
            class_balance=None,
            model_class_balance=None,
            model_propensity=None,
            model_accuracy=None,
            propensity_range=(1.0, 1.0),
            accuracy_range=(0.7, 0.9),
            overlap_factor=0.5,
            subsuming_factor=0.2,
            random_state=12345,
            **kwargs
    ):
        self.m = len(model_classes)
        self.k = len(desired_classes)
        self.model_classes = model_classes
        self.desired_classes = desired_classes
        self.undesired_classes = list(set(chain.from_iterable(model_classes)))
        self.n_classes = len(set(chain.from_iterable(model_classes)).union(desired_classes))
        self.label_relations = label_relations

        # initialize random state
        self.generator = check_random_state(random_state)
        self.overlap_factor = overlap_factor
        self.subsuming_factor = subsuming_factor

        # init label graph with EXCLUSIVE relation
        self.label_graph = np.ones((self.n_classes, self.n_classes), dtype=int) * EXCLUSIVE
        for i, j, v in label_relations:
            self.label_graph[j, i] = reverse_relation(v)
            self.label_graph[i, j] = v

        # for mutual exclusive assumption
        for classes in self.model_classes:
            for i, j in permutations(classes, 2):
                self.label_graph[i][j] = EXCLUSIVE
        for i, j in permutations(desired_classes, 2):
            self.label_graph[i][j] = EXCLUSIVE

        # self-loop
        # np.fill_diagonal(self.label_graph, 1)
        np.fill_diagonal(self.label_graph, EQUAL)

        # p(seen labels | desired labels), each row does not sum to 1!
        # assume only 4 types of relations!
        self.cond_prob = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            for j in range(self.k, self.n_classes):
                if self.label_graph[i][j] == INCLUDED:
                    self.cond_prob[i, j] = 1
                elif self.label_graph[i][j] == INCLUSION:
                    self.cond_prob[i, j] = self.generator.uniform(0, subsuming_factor)
                elif self.label_graph[i][j] == OVERLAP:
                    self.cond_prob[i, j] = self.generator.uniform(0, overlap_factor)

        # Generate class balance self.p
        if class_balance is None:
            self.p = np.full(self.k, 1 / self.k)
        else:
            self.p = class_balance

        if model_class_balance is None:
            self.model_p = [np.full(len(label_space), 1 / len(label_space)) for label_space in model_classes]
        else:
            self.model_p = model_class_balance

        if model_propensity is None:
            self.model_propensity = self.generator.uniform(propensity_range[0], propensity_range[1], self.m)
        else:
            self.model_propensity = model_propensity

        if model_accuracy is None:
            self.model_accuracy = self.generator.uniform(accuracy_range[0], accuracy_range[1], self.m)
        else:
            self.model_accuracy = model_accuracy

    def check_exclusive(self, y, ys):
        m = self.label_graph[y, ys]
        return np.all(m != EXCLUSIVE)

    def complete_ys(self, y, ys):
        ys_ = ys.copy()
        for y_u in self.generator.permutation(ys):
            for yy_u in self.generator.permutation(self.undesired_classes):
                if yy_u not in ys_ and self.check_exclusive(yy_u, ys_ + [y]):
                    p = self.cond_prob[y_u, yy_u]
                    if p > 0 and (self.generator.random() < p):
                        ys_.append(yy_u)

        return ys_

    def generate(self, n):
        Y = self.generator.choice(self.desired_classes, n, p=self.p)
        Ys = []
        for y in Y:
            ys = []
            for y_u in self.generator.permutation(self.undesired_classes):
                p = self.cond_prob[y, y_u]
                if p > 0 and (self.generator.random() < p) and self.check_exclusive(y_u, ys):
                    ys.append(y_u)
            ys = self.complete_ys(y, ys)
            Ys.append(set(ys + [y]))

        Ls = []
        for model_id, label_space in enumerate(self.model_classes):
            propensity = self.model_propensity[model_id]
            accuracy = self.model_accuracy[model_id]
            model_balance = self.model_p[model_id]
            L_i = []
            for ys in Ys:
                if self.generator.random() < propensity:
                    inter_labels = [i for i in ys if i in label_space]
                    if len(inter_labels) == 0:
                        chosen = self.generator.choice(label_space, p=model_balance)
                        L_i.append(chosen)
                    elif len(inter_labels) == 1:
                        correct = inter_labels[0]
                        if self.generator.random() < accuracy:
                            L_i.append(correct)
                        else:
                            chosen = self.choose_other_label(label_space, correct)
                            L_i.append(chosen)
                    else:
                        raise NotImplementedError
                else:
                    L_i.append(ABSTAIN)
            Ls.append(L_i)

        Ls = np.array(Ls).transpose()

        return SyntheticDataset(
            L=Ls,
            y=Y,
            classes=self.desired_classes,
            model_class=self.model_classes,
            label_relations=self.label_relations
        )

    def choose_other_label(self, classes, y):
        """Given a cardinality k and true label y, return random value in
        {1,...,k} \ {y}."""
        return self.generator.choice([i for i in classes if i != y])
