from collections import defaultdict, Counter

import numpy as np
import scipy as sp

from ..label import INCLUDED, EXCLUSIVE, ABSTAIN, reverse_relation


def to_one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def desired_majority_voting(Ls, desired_class):
    res = []
    for L in Ls:
        L_ = [i for i in L if i in desired_class]
        if len(L_) == 0:
            res.append(np.random.choice(desired_class))
        else:
            res.append(sp.stats.mode(L_)[0])
    return np.array(res, dtype=int)


def label_relation_majority_voting(Ls, desired_class, mappings, return_proba=False):
    res = []
    mapping = {}
    for m, n in mappings:
        if n in desired_class and m not in desired_class:
            if m not in mapping:
                mapping[m] = [n]
            else:
                mapping[m].append(n)
        elif m in desired_class and n not in desired_class:
            if n not in mapping:
                mapping[n] = [m]
            else:
                mapping[n].append(m)

    for L in Ls:
        L_ = []
        for i in L:
            if i in desired_class:
                L_.append(i)
            elif i != ABSTAIN:
                L_.extend(mapping[i])
        if len(L_) == 0:
            res.append(np.random.choice(desired_class))
        else:
            counter = Counter(L_)
            max_votes = max(list(counter.values()))
            max_labels = [i for i, v in counter.items() if max_votes == v]
            res.append(np.random.choice(max_labels))

    res = np.array(res, dtype=int)
    if return_proba:
        res = to_one_hot(res, len(desired_class))
    return res


def advanced_label_relation_majority_voting(Ls, desired_class, label_relation, return_proba=False):
    res = []
    mapping = {}
    for m, n, r in label_relation:
        if r != EXCLUSIVE:
            if n in desired_class and m not in desired_class:
                if m not in mapping:
                    mapping[m] = [(n, r)]
                else:
                    mapping[m].append((n, r))
            elif m in desired_class and n not in desired_class:
                if n not in mapping:
                    mapping[n] = [(m, reverse_relation(r))]
                else:
                    mapping[n].append((m, reverse_relation(r)))

    weighted_mapping = defaultdict(list)
    for m, ns in mapping.items():
        cnt = 0
        for n, r in ns:
            if r == INCLUDED:
                weighted_mapping[m].append((n, 1))
            else:
                cnt += 1
        for n, r in ns:
            if r != INCLUDED:
                weighted_mapping[m].append((n, 1 / cnt))

    for L in Ls:
        L_ = {i: 0 for i in desired_class}
        for i in L:
            if i in desired_class:
                L_[i] += 1
            elif i != ABSTAIN:
                for n, w in weighted_mapping[i]:
                    L_[n] += w

        max_votes = max(list(L_.values()))
        max_labels = [i for i, v in L_.items() if max_votes == v]
        res.append(np.random.choice(max_labels))

    res = np.array(res, dtype=int)
    if return_proba:
        res = to_one_hot(res, len(desired_class))
    return res
