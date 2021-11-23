from collections import defaultdict
from itertools import chain
from numbers import Integral
from typing import List, Tuple, Union, Iterable

import networkx as nx
import numpy as np
from tqdm import trange, tqdm

from .core import *
from ..utils import check_random_state

LabelRelationConstraints = {
    (OVERLAP, INCLUDED)   : [INCLUDED, OVERLAP],
    (OVERLAP, INCLUSION)  : [EXCLUSIVE, INCLUSION, OVERLAP],
    (OVERLAP, EXCLUSIVE)  : [EXCLUSIVE, INCLUSION, OVERLAP],
    (OVERLAP, OVERLAP)    : [EXCLUSIVE, INCLUSION, INCLUDED, OVERLAP],
    (INCLUSION, EXCLUSIVE): [EXCLUSIVE, INCLUSION, OVERLAP],
    (INCLUSION, INCLUSION): [INCLUSION],
    (INCLUSION, OVERLAP)  : [INCLUSION, OVERLAP],
    (INCLUSION, INCLUDED) : [INCLUDED, INCLUSION, OVERLAP],
    (EXCLUSIVE, INCLUDED) : [INCLUDED, EXCLUSIVE, OVERLAP],
    (EXCLUSIVE, INCLUSION): [EXCLUSIVE],
    (EXCLUSIVE, OVERLAP)  : [INCLUDED, EXCLUSIVE, OVERLAP],
    (EXCLUSIVE, EXCLUSIVE): [EXCLUSIVE, INCLUSION, INCLUDED, OVERLAP],
    (INCLUDED, INCLUSION) : [EXCLUSIVE, INCLUSION, INCLUDED, OVERLAP],
    (INCLUDED, OVERLAP)   : [INCLUDED, EXCLUSIVE, OVERLAP],
    (INCLUDED, INCLUDED)  : [INCLUDED],
    (INCLUDED, EXCLUSIVE) : [EXCLUSIVE],
}

LabelRelationConstraintsForbidden = [
    (OVERLAP, INCLUDED, INCLUSION),
    (OVERLAP, INCLUDED, EXCLUSIVE),
    (OVERLAP, INCLUSION, INCLUDED),
    (OVERLAP, EXCLUSIVE, INCLUDED),

    (EXCLUSIVE, INCLUDED, INCLUSION),
    (EXCLUSIVE, OVERLAP, INCLUSION),
    (EXCLUSIVE, INCLUSION, INCLUSION),
    (EXCLUSIVE, INCLUSION, INCLUDED),
    (EXCLUSIVE, INCLUSION, OVERLAP),

    (INCLUSION, EXCLUSIVE, INCLUDED),
    (INCLUSION, INCLUDED, EXCLUSIVE),
    (INCLUSION, OVERLAP, INCLUDED),
    (INCLUSION, OVERLAP, EXCLUSIVE),
    (INCLUSION, INCLUSION, EXCLUSIVE),
    (INCLUSION, INCLUSION, INCLUDED),
    (INCLUSION, INCLUSION, OVERLAP),

    (INCLUDED, OVERLAP, INCLUSION),
    (INCLUDED, INCLUDED, EXCLUSIVE),
    (INCLUDED, INCLUDED, INCLUSION),
    (INCLUDED, INCLUDED, OVERLAP),
    (INCLUDED, EXCLUSIVE, INCLUSION),
    (INCLUDED, EXCLUSIVE, INCLUDED),
    (INCLUDED, EXCLUSIVE, OVERLAP),
]

relations = [EXCLUSIVE, OVERLAP, INCLUSION, INCLUDED]
LabelRelationConstraintsAllowed = set()
for i in relations:
    for j in relations:
        for t in relations:
            if (i, j, t) not in LabelRelationConstraintsForbidden:
                LabelRelationConstraintsAllowed.add((i, j, t))


def generate_allowed_label_relations(rs1: Iterable, rs2: Iterable):
    allowed = set()
    for r1 in rs1:
        for r2 in rs2:
            allowed.update(LabelRelationConstraints[(r1, r2)])
    return allowed


class LabelRelation:
    def __init__(self, label_relations: List[Tuple]) -> None:
        self.labels = set()
        self.label_relations = {}
        self.label_relation_list = []
        self.constraints = {}
        self.label_graph = defaultdict(list)
        self.add_relations(label_relations)

    def get_label_relations(self) -> List[Tuple]:
        return self.label_relation_list

    def get_desired_mappings(self, desired_classes: List, relation_values: List[Union[str, Integral]] = [EQUAL, OVERLAP, INCLUDED, INCLUSION]) -> List[Tuple]:
        maps = []
        for m, n, v in self.label_relation_list:
            if v in relation_values:
                if n in desired_classes and m not in desired_classes:
                    maps.append((m, n))
                elif m in desired_classes and n not in desired_classes:
                    maps.append((n, m))
        return maps

    def update_constraints(self, label_pair: Tuple, relation: Integral):
        m, n = label_pair
        reversed_relation = reverse_relation(relation)
        for n1, r in self.label_graph[n]:
            if (m, n1) in self.constraints:
                self.constraints[(m, n1)].intersection_update(LabelRelationConstraints[(relation, r)])
            if (n1, m) in self.constraints:
                self.constraints[(n1, m)].intersection_update(LabelRelationConstraints[(reverse_relation(r), reversed_relation)])
        for m1, r in self.label_graph[m]:
            if (n, m1) in self.constraints:
                self.constraints[(n, m1)].intersection_update(LabelRelationConstraints[(reversed_relation, r)])
            if (m1, n) in self.constraints:
                self.constraints[(m1, n)].intersection_update(LabelRelationConstraints[(reverse_relation(r), relation)])

        if (m, n) in self.constraints:
            del self.constraints[m, n]
        if (n, m) in self.constraints:
            del self.constraints[n, m]

    def check_relation_exist(self, m, n):
        return (m, n) in self.label_relations or (n, m) in self.label_relations

    def update_label_relations(self, m, n, relation_value) -> None:
        key = (m, n)

        n_labels = len(self.labels)
        self.labels.add(m)
        new_m = len(self.labels) > n_labels
        self.labels.add(n)
        new_n = len(self.labels) > n_labels

        if len(self.labels) == n_labels:
            if self.check_relation_exist(m, n):
                raise ValueError(f'label relation {key} already exists, try to add {relation_value}!')
            if ((m, n) in self.constraints and relation_value not in self.constraints[(m, n)]):
                raise ValueError(f'label pair {key} and relation {relation_value} is invalid!')

        self.label_relations[key] = relation_value
        self.label_relation_list.append((m, n, relation_value))

        if new_m:
            for i in self.labels:
                if (not self.check_relation_exist(i, m)) and i != m:
                    self.constraints[(i, m)] = set([OVERLAP, EXCLUSIVE, INCLUSION, INCLUDED])
                    self.constraints[(m, i)] = set([OVERLAP, EXCLUSIVE, INCLUSION, INCLUDED])
        if new_n:
            for i in self.labels:
                if (not self.check_relation_exist(i, n)) and i != n:
                    self.constraints[(i, n)] = set([OVERLAP, EXCLUSIVE, INCLUSION, INCLUDED])
                    self.constraints[(n, i)] = set([OVERLAP, EXCLUSIVE, INCLUSION, INCLUDED])

        self.update_constraints(key, relation_value)
        self.label_graph[m].append((n, relation_value))
        self.label_graph[n].append((m, reverse_relation(relation_value)))

    def add_relations(self, label_relations: List[Tuple]) -> None:
        for label_relation in label_relations:
            self.add_relation(label_relation)

    def add_relation(self, label_relation: Tuple) -> None:
        m, n, v = label_relation
        v = relation_value(v)
        assert v in [OVERLAP, EXCLUSIVE, INCLUSION, INCLUDED]
        self.update_label_relations(m, n, v)

    def build_label_graph(self):
        labels = list(self.labels)
        label_graph = np.ones((len(labels), len(labels))) * UNKNOWN
        for m, n, v in self.label_relation_list:
            i, j = labels.index(m), labels.index(n)
            label_graph[i, j] = v
            label_graph[j, i] = reverse_relation(v)
        return labels, label_graph

    def __iter__(self):
        return iter(self.label_relation_list)


MAX_OVERLAP_CNT = 2


class LabelGraphGenerator:
    def __init__(self, n_desired_labels, n_all_labels, random_state=12345) -> None:
        # initialize random state
        self.generator = check_random_state(random_state)

        self.n_desired_labels = n_desired_labels
        self.n_undesired_labels = n_all_labels - n_desired_labels
        self.n_all_labels = n_all_labels

    @staticmethod
    def _is_subclique(G, nodelist):
        H = G.subgraph(nodelist)
        n = len(nodelist)
        return H.size() == n * (n - 1) / 2

    def _sample_exclusive_connected_subgraph(self, graph, n_node, n_subgraph, force_cover=False, tol=100):
        nodes = graph.nodes
        n = len(nodes)
        if isinstance(n_node, list):
            assert n >= max(n_node)
        else:
            assert n >= n_node
            n_node = [n_node]

        if force_cover:
            min_n_neighbor = min(n_node)
            for node in nodes:
                if graph.degree(node) < min_n_neighbor:
                    return []

        n_node_to_candidates = {ni: [i for i in nodes if graph.degree(i) >= (ni - 1)] for ni in n_node}
        subgraphs = []
        i = 0
        while i < tol:
            i += 1
            graph_size = self.generator.choice(n_node)
            candidates = n_node_to_candidates[graph_size]
            if len(candidates) < graph_size:
                return []
            subgraph = self.generator.choice(candidates, graph_size, replace=False)
            if self._is_subclique(graph, subgraph):
                subgraphs.append(subgraph)
                if len(subgraphs) == n_subgraph:
                    if force_cover:
                        covered_nodes = set(chain.from_iterable(subgraphs))
                        if len(covered_nodes) == n:
                            return subgraphs
                        else:
                            subgraphs = []
                    else:
                        return subgraphs

        return []

    def _sample_non_neighbors(self, label_relation_matrix, graph, nodes, n_neighbor, tol=10):
        node_to_non_neighbors = {}
        for n in nodes:
            non_neighbors = list(nx.non_neighbors(graph, n))
            if len(non_neighbors) == 0:
                return []
            node_to_non_neighbors[n] = non_neighbors

        cnt = 0
        while cnt < tol:
            cnt += 1
            seen = set()
            while len(seen) < n_neighbor:
                perm = self.generator.permutation(len(nodes))
                for n in nodes[perm]:
                    seen.add(self.generator.choice(node_to_non_neighbors[n]))
                    if len(seen) == n_neighbor:
                        break

            seen = list(seen)
            if self._check_unseen_valid(label_relation_matrix, nodes, seen):
                return seen

        return []

    @staticmethod
    def _check_unseen_valid(label_relation_matrix, unseen_labels, seen_labels):
        m = label_relation_matrix[np.ix_(unseen_labels, seen_labels)]

        if np.sum(m == OVERLAP) > (np.sum(m != EXCLUSIVE) / 2):
            return False

        single_cnt = 0
        for i, v in enumerate(m):
            single_flag = True
            for j, vv in enumerate(v):
                if vv != EXCLUSIVE and np.sum(m[:, j] != EXCLUSIVE) > 1:
                    single_flag = False
                    break
            if single_flag:
                single_cnt += 1

        if single_cnt == len(unseen_labels):
            return False

        for i in range(len(seen_labels)):
            if np.all(m[:, i] == m[0, i]):
                return False

        for i, v in enumerate(m):
            non_exclusive_neighbor_cnt = np.sum(v != EXCLUSIVE)
            if non_exclusive_neighbor_cnt < 1:
                # uncovered node
                return False
            else:
                for j in range(i + 1, len(unseen_labels)):
                    if np.all(v == m[j]):
                        return False

        return True

    @staticmethod
    def _convert_label(desired_labels, all_labels, all_model_labels, label_relations):
        label_to_id = {}
        for l in desired_labels:
            label_to_id[l] = len(label_to_id)
        for l in all_labels:
            if l not in label_to_id:
                label_to_id[l] = len(label_to_id)

        desired_labels = [label_to_id[l] for l in desired_labels]
        all_labels = sorted([label_to_id[l] for l in all_labels])
        all_model_labels = [sorted([label_to_id[l] for l in ls]) for ls in all_model_labels]
        label_relations = [(label_to_id[m], label_to_id[n], r) for m, n, r in label_relations]

        return desired_labels, all_labels, all_model_labels, label_relations

    def _sample(self,
                label_relation_matrix,
                exclusive_graph,
                n_models,
                model_label_space_lo,
                model_label_space_hi,
                tol=1000,
                ):

        desired_labels = self._sample_exclusive_connected_subgraph(exclusive_graph, self.n_desired_labels, 1, tol=tol)
        if len(desired_labels) == 0:
            return None
        desired_labels = desired_labels[0]

        seen_labels = self._sample_non_neighbors(
            label_relation_matrix,
            exclusive_graph,
            desired_labels,
            self.n_undesired_labels
        )
        if len(seen_labels) == 0:
            return None

        all_model_labels = self._sample_exclusive_connected_subgraph(
            exclusive_graph.subgraph(seen_labels),
            list(range(model_label_space_lo, model_label_space_hi + 1)),
            n_models,
            force_cover=True,
            tol=tol
        )
        if len(all_model_labels) == 0:
            return None

        all_labels = list(seen_labels) + list(desired_labels)
        label_relations = []
        for i in range(self.n_all_labels):
            li = all_labels[i]
            for j in range(i + 1, self.n_all_labels):
                lj = all_labels[j]
                r = label_relation_matrix[li, lj]
                label_relations.append((li, lj, r))

        return desired_labels, all_labels, all_model_labels, label_relations

    def _sample_n(self,
                  label_relation_matrix,
                  exclusive_graph,
                  n_models,
                  model_label_space_lo,
                  model_label_space_hi,
                  n_samples,
                  tol=100,
                  ):
        sampled_label_relations = []
        sampled_model_labels = []
        sampled_desired_labels = []
        sampled_all_labels = []

        for _ in trange(n_samples):
            while True:
                results = self._sample(
                    label_relation_matrix,
                    exclusive_graph,
                    n_models,
                    model_label_space_lo,
                    model_label_space_hi,
                    tol=tol
                )
                if results is not None:
                    desired_labels, all_labels, all_model_labels, label_relations = results
                    break

            desired_labels, all_labels, all_model_labels, label_relations = \
                self._convert_label(desired_labels, all_labels, all_model_labels, label_relations)

            sampled_label_relations.append(label_relations)
            sampled_model_labels.append(all_model_labels)
            sampled_desired_labels.append(desired_labels)
            sampled_all_labels.append(all_labels)

        return sampled_label_relations, sampled_model_labels, sampled_desired_labels, sampled_all_labels

    def generate_tree(self,
                      n_models,
                      model_label_space_lo,
                      model_label_space_hi,
                      n_samples,
                      r=3,
                      h=4,
                      tol=100,
                      ):

        tree = nx.bfs_tree(nx.balanced_tree(r=r, h=h), 0)
        nodes = list(tree.nodes)
        n = len(nodes)

        label_relation_matrix = np.ones((n, n), dtype=int) * EXCLUSIVE
        for node in nodes:
            for c in nx.descendants(tree, node):
                label_relation_matrix[node, c] = INCLUSION
                label_relation_matrix[c, node] = INCLUDED
        np.fill_diagonal(label_relation_matrix, EQUAL)

        exclusive_graph = nx.Graph()
        for i in range(n):
            ni = nodes[i]
            for j in range(i + 1, n):
                nj = nodes[j]
                if label_relation_matrix[ni, nj] == EXCLUSIVE:
                    exclusive_graph.add_edge(ni, nj)

        return self._sample_n(label_relation_matrix,
                              exclusive_graph,
                              n_models,
                              model_label_space_lo,
                              model_label_space_hi,
                              n_samples,
                              tol=tol,
                              )

    @staticmethod
    def _check_overlap(graph, node, des):
        for n in nx.descendants(graph, node):
            if n in des:
                return True
        return False

    def generate_dag(self,
                     n_models,
                     model_label_space_lo,
                     model_label_space_hi,
                     n_samples,
                     graph_size=200,
                     p=0.1,
                     tol=100,
                     ):

        max_indegree = 2
        random_graph = nx.gnp_random_graph(graph_size, p, directed=True)
        dag = nx.DiGraph([(u, v) for (u, v) in random_graph.edges() if u < v])

        outer_flag = True
        while outer_flag:
            outer_flag = False

            for n, d in dict(dag.in_degree).items():
                if d > max_indegree:
                    ps = self.generator.choice(list(dag.predecessors(n)), d - max_indegree, replace=False)
                    for p in ps:
                        dag.remove_edge(p, n)

            flag = True
            while flag:
                flag = False
                node_topo = list(nx.topological_sort(dag))
                for n in tqdm(reversed(node_topo), desc='delete useless nodes...'):
                    cs = list(dag.successors(n))
                    if len(cs) == 1:
                        flag = True
                        dag = nx.contracted_nodes(dag, cs[0], n, self_loops=False)

            for cycle in tqdm(nx.cycle_basis(dag.to_undirected(as_view=True)), desc='delete jump edges...'):
                subgraph = dag.subgraph(cycle)
                root = [n for n in subgraph if subgraph.in_degree(n) == 0]
                if len(root) == 1:
                    leave = [n for n in subgraph if subgraph.out_degree(n) == 0]
                    if len(leave) == 1:
                        root = root[0]
                        leave = leave[0]
                        if dag.has_edge(root, leave):
                            outer_flag = True
                            dag.remove_edge(root, leave)

        dag = nx.relabel_nodes(dag, dict(zip(list(dag.nodes), range(len(dag)))))

        nodes = list(dag.nodes)
        n = len(nodes)
        label_relation_matrix = np.ones((n, n), dtype=int) * EXCLUSIVE
        for i in trange(n):
            ni = nodes[i]
            des = nx.descendants(dag, ni)
            for j in range(i + 1, n):
                nj = nodes[j]
                if nj in des:
                    label_relation_matrix[ni, nj] = INCLUSION
                    label_relation_matrix[nj, ni] = INCLUDED
                elif self._check_overlap(dag, nj, des):
                    label_relation_matrix[ni, nj] = OVERLAP
                    label_relation_matrix[nj, ni] = OVERLAP
        np.fill_diagonal(label_relation_matrix, EQUAL)

        exclusive_graph = nx.Graph()
        for i in range(n):
            ni = nodes[i]
            for j in range(i + 1, n):
                nj = nodes[j]
                if label_relation_matrix[ni, nj] == EXCLUSIVE:
                    exclusive_graph.add_edge(ni, nj)

        return self._sample_n(label_relation_matrix,
                              exclusive_graph,
                              n_models,
                              model_label_space_lo,
                              model_label_space_hi,
                              n_samples,
                              tol=tol,
                              )
