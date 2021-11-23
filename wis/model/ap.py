from collections import Counter
from copy import deepcopy
from itertools import permutations, chain

import numpy as np

from ..label import EQUAL, INCLUDED, EXCLUSIVE, ABSTAIN, reverse_relation


def build_attribute(desired_class, model_classes, label_relations):
    all_class = [i for i in set(chain.from_iterable(model_classes)).union(desired_class) if i != ABSTAIN]
    undesired_class = [i for i in all_class if i not in desired_class]

    label_graph = np.ones((len(all_class), len(all_class)), dtype=int) * EXCLUSIVE
    for i, j, v in label_relations:
        label_graph[j, i] = reverse_relation(v)
        label_graph[i, j] = v

    # for mutual exclusive assumption
    for classes in model_classes:
        for i, j in permutations(classes, 2):
            label_graph[i][j] = EXCLUSIVE
    for i, j in permutations(desired_class, 2):
        label_graph[i][j] = EXCLUSIVE

    # self-loop
    # np.fill_diagonal(self.label_graph, 1)
    np.fill_diagonal(label_graph, EQUAL)

    # flag: 0-could, 1-not, 2-must
    possible_assignments = []
    for y_d in desired_class:
        init = [[y_d]]
        for y_u in undesired_class:
            if label_graph[y_u, y_d] != EXCLUSIVE:
                cur = deepcopy(init)
                for i, assignment in enumerate(cur):
                    flag = 0
                    for y in assignment:
                        r = label_graph[y, y_u]
                        if r == EXCLUSIVE or r == EQUAL:
                            flag = 1
                            break
                        if r == INCLUDED:
                            flag = 2
                            break
                    if flag == 2:
                        init[i].append(y_u)
                    if flag == 0:
                        init.append(assignment + [y_u])
        possible_assignments.extend(init)

    attributes = []
    desired_class_to_attributes = {k: [] for k in desired_class}
    for a in possible_assignments:
        for y_d in desired_class:
            if y_d in a:
                aa = tuple([i for i in a if i not in desired_class])
                if len(aa) > 0:
                    if aa in attributes:
                        desired_class_to_attributes[y_d].append(attributes.index(aa))
                    else:
                        desired_class_to_attributes[y_d].append(len(attributes))
                        attributes.append(aa)

    desired_class_attribute_matrix = np.zeros((len(desired_class), len(attributes)))
    for y_d, aa in desired_class_to_attributes.items():
        desired_class_attribute_matrix[y_d, aa] = 1

    return desired_class_attribute_matrix, attributes


def generate_attribute_vector(dic, attributes):
    unique_labels = tuple([i for i, v in dic.items() if i != -1 and v > 0])
    av = np.zeros(len(attributes))
    for i, a in enumerate(attributes):
        if a == unique_labels:
            av[i] = 1
    return av


def aggregate_ilf_to_attribute(L_mv, attributes):
    n, m = L_mv.shape
    A = np.zeros((n, len(attributes)))

    for i, l in enumerate(L_mv):
        counter = Counter(l)
        A[i] = generate_attribute_vector(counter, attributes)

    return A
