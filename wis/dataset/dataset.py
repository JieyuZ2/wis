import pickle
import time
from abc import abstractmethod
from itertools import chain
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm, trange

from ..label import EQUAL, OVERLAP, INCLUSION, INCLUDED, EXCLUSIVE, ABSTAIN
from ..mytype import Config
from ..utils import check_random_state


class DatasetConfig(Config):
    n_data_source_each_class: int = 500
    n_data_target_each_class: int = 1000
    n_samples: int = 1
    n_desired_labels: int = 5
    n_labels: int = 10

    n_models_per_label: int = 3

    n_models: int = 10
    model_label_space_lo: int = 2
    model_label_space_hi: int = 3

    seed: int = 12345


def reverse_dict(d):
    return {i: k for k, vs in d.items() for i in vs}


class HierDataset:
    def __init__(self, dataset, label_min_support=1000, random_state=12345, preprocessed_path=None):
        start = time.time()
        self.generator = check_random_state(random_state)

        load_preprocessed = False
        if preprocessed_path is not None:
            preprocessed_path = Path(preprocessed_path)
            if preprocessed_path.exists():
                print(f'load preprocessed data from {preprocessed_path}')
                preprocessed = pickle.load(open(preprocessed_path, 'rb'))
                load_preprocessed = True
        if load_preprocessed:
            self.graph: nx.DiGraph = preprocessed['hier']
            self.labels = np.array(preprocessed['labels'])
            self.class_to_data_idx = preprocessed['class_to_data_idx']
        else:
            self.graph: nx.DiGraph = dataset['hier']
            self.labels = np.array(dataset['labels'])
            self.class_to_data_idx = self._init_dataset()
            self._init_graph()

        self.label_relation_cache = {}
        self.graph_undirected_view = self.graph.to_undirected(as_view=True)

        if load_preprocessed and 'exclusive_graph' in preprocessed:
            self.exclusive_graph = preprocessed['exclusive_graph']
            self.candidate_labels = list(self.exclusive_graph.nodes)
        else:
            self.candidate_labels = [n for n in self.graph.nodes if len(self.class_to_data_idx[n]) > label_min_support and self.graph.in_degree(n) > 0]
            n = len(self.candidate_labels)
            graph = nx.Graph()
            for i in trange(n):
                ni = self.candidate_labels[i]
                for j in range(i + 1, n):
                    nj = self.candidate_labels[j]
                    if self.get_relation(ni, nj) == EXCLUSIVE:
                        graph.add_edge(ni, nj)
            self.exclusive_graph = graph

        end = time.time()
        print(f"load dataset in {end - start} seconds")

    def __len__(self):
        return len(self.labels)

    def _init_dataset(self):
        class_to_data_idx = {}
        for i, class_name in tqdm(enumerate(self.labels), desc='build dataset'):
            if class_name not in class_to_data_idx:
                class_to_data_idx[class_name] = []
            class_to_data_idx[class_name].append(i)
        return class_to_data_idx

    def _init_graph(self):
        """
        Aggregate the data in DiGraph
        Topological Sort and aggregate in reversed order
        """
        node_topo = list(nx.topological_sort(self.graph))
        for n in tqdm(reversed(node_topo), desc='aggregate data from children to parent node and delete node without data...'):
            children_labels = set()
            for c in self.graph.successors(n):
                children_labels.update(self.class_to_data_idx[c])
            self_labels = set(self.class_to_data_idx.get(n, []))
            all_labels = list(children_labels.union(self_labels))
            if len(all_labels) == 0:
                self.graph.remove_node(n)
            else:
                self.class_to_data_idx[n] = all_labels

        outer_flag = True
        while outer_flag:
            outer_flag = False

            flag = True
            while flag:
                flag = False
                node_topo = list(nx.topological_sort(self.graph))
                for n in tqdm(reversed(node_topo), desc='delete useless nodes...'):
                    cs = list(self.graph.successors(n))
                    if len(cs) == 1:
                        flag = True
                        self.graph = nx.contracted_nodes(self.graph, cs[0], n, self_loops=False)

            for cycle in tqdm(nx.cycle_basis(self.graph.to_undirected(as_view=True)), desc='delete jump edges...'):
                subgraph = self.graph.subgraph(cycle)
                root = [n for n in subgraph if subgraph.in_degree(n) == 0]
                if len(root) == 1:
                    leave = [n for n in subgraph if subgraph.out_degree(n) == 0]
                    if len(leave) == 1:
                        root = root[0]
                        leave = leave[0]
                        if self.graph.has_edge(root, leave):
                            outer_flag = True
                            self.graph.remove_edge(root, leave)

        self.class_to_data_idx = {n: self.class_to_data_idx[n] for n in self.graph}

    def _get_class_data(self, class_name, sample_number=-1, existing_instances=None):
        """

        Args:
            class_name: label / class name
            sample_number: number of data to sample from this class
            existing_instances: set of indexes not sampled from this class

        Returns:
            list of indexes for sampled data
        """
        assert class_name in self.class_to_data_idx, f"{class_name} not found in taxonomy"
        data_idx = self.class_to_data_idx[class_name].copy()
        if existing_instances is not None:
            tmp = set(data_idx) - existing_instances
            if len(tmp) != len(data_idx):
                data_idx = list(tmp)
                print(f"Warning: {class_name} caused overlap")

        if sample_number >= 0:
            max_cnt = len(data_idx)
            if sample_number > max_cnt:
                print(f"sample number for class {class_name} exceeds available limit, return ({max_cnt}) data instead")
            if sample_number < max_cnt:
                data_idx = self.generator.choice(data_idx, sample_number, replace=False)

        return np.array(data_idx)

    def _get_classes_data(self, classes_name, sample_number=-1, disjoint=True, exclude_idx=[]):
        class_to_data_idx = {}
        exclude_idx = set(exclude_idx)
        for class_name in classes_name:
            class_to_data_idx[class_name] = self._get_class_data(
                class_name=class_name, sample_number=sample_number, existing_instances=exclude_idx)
            if disjoint:
                exclude_idx.update(class_to_data_idx[class_name])
        instance_idx = list(chain.from_iterable(class_to_data_idx.values()))
        return class_to_data_idx, instance_idx

    @abstractmethod
    def get_dataset(self, classes_name, sample_number=-1, disjoint=True):
        pass

    def _get_nonclass_data(self, class_name, sample_number=-1, existing_instances=None):
        """

        Args:
            class_name: label / class name
            sample_number: number of data to sample from this class
            existing_instances: set of indexes not sampled from this class

        Returns:
            list of indexes for sampled data
        """
        assert class_name in self.class_to_data_idx, f"{class_name} not found in taxonomy"
        data_idx_to_avoid = set(self.class_to_data_idx[class_name])
        if existing_instances is not None:
            data_idx = [i for i in range(len(self)) if i not in data_idx_to_avoid and i not in existing_instances]
        else:
            data_idx = [i for i in range(len(self)) if i not in data_idx_to_avoid]

        if sample_number >= 0:
            max_cnt = len(data_idx)
            if sample_number > max_cnt:
                print(f"sample number for class {class_name} exceeds available limit, return ({max_cnt}) data instead")
            if sample_number < max_cnt:
                data_idx = self.generator.choice(data_idx, sample_number, replace=False)

        return np.array(data_idx)

    def _get_binary_data(self, class_name, sample_number=-1, exclude_idx=[]):
        class_to_data_idx = {}
        exclude_idx = set(exclude_idx)

        class_to_data_idx[class_name] = self._get_class_data(
            class_name=class_name, sample_number=sample_number, existing_instances=exclude_idx)

        class_to_data_idx[ABSTAIN] = self._get_nonclass_data(
            class_name=class_name, sample_number=sample_number, existing_instances=exclude_idx)

        instance_idx = list(chain.from_iterable(class_to_data_idx.values()))
        return class_to_data_idx, instance_idx

    @abstractmethod
    def get_binary_dataset(self, class_name, sample_number=-1):
        pass

    def get_relation(self, c1, c2):
        if (c1, c2) in self.label_relation_cache:
            return self.label_relation_cache[(c1, c2)]
        else:
            c1_data = set(self.class_to_data_idx[c1])
            c2_data = set(self.class_to_data_idx[c2])
            inter_data = c1_data & c2_data
            len_c1_data = len(c1_data)
            len_c2_data = len(c2_data)
            assert len_c1_data > 0 and len_c2_data > 0
            len_inter_data = len(inter_data)
            if len_inter_data == 0:
                self.label_relation_cache[(c1, c2)] = EXCLUSIVE
                self.label_relation_cache[(c2, c1)] = EXCLUSIVE
                return EXCLUSIVE
            else:
                if len_inter_data == len_c1_data == len_c2_data:
                    self.label_relation_cache[(c1, c2)] = EQUAL
                    self.label_relation_cache[(c2, c1)] = EQUAL
                    return EQUAL
                elif len_inter_data == len_c1_data < len_c2_data:
                    self.label_relation_cache[(c1, c2)] = INCLUDED
                    self.label_relation_cache[(c2, c1)] = INCLUSION
                    return INCLUDED
                elif len_inter_data == len_c2_data < len_c1_data:
                    self.label_relation_cache[(c1, c2)] = INCLUSION
                    self.label_relation_cache[(c2, c1)] = INCLUDED
                    return INCLUSION
                else:
                    assert len_inter_data < len_c1_data and len_inter_data < len_c2_data
                    self.label_relation_cache[(c1, c2)] = OVERLAP
                    self.label_relation_cache[(c2, c1)] = OVERLAP
                    return OVERLAP

    def _check_unseen_valid(self, unseen_labels, seen_labels):
        n_seen = len(seen_labels)
        m = np.zeros((len(unseen_labels), n_seen))
        for i, l in enumerate(unseen_labels):
            for j, l1 in enumerate(seen_labels):
                m[i, j] = self.get_relation(l, l1)

        m_seen = np.zeros((n_seen, n_seen))
        for i, l in enumerate(seen_labels):
            for j, l1 in enumerate(seen_labels):
                if i == j:
                    m_seen[i, i] = EXCLUSIVE
                else:
                    m_seen[i, j] = self.get_relation(l, l1)

        # if np.sum(m==OVERLAP) > 0 or np.sum(m_seen==OVERLAP) > 0:
        #     return False

        single_cnt = 0
        for i, v in enumerate(m):
            single_flag = True
            for j, vv in enumerate(v):
                if vv != EXCLUSIVE and np.sum(m[:, j] != EXCLUSIVE) > 1:
                    single_flag = False
                    break
            if single_flag:
                single_cnt += 1
                # return False

        if single_cnt == len(unseen_labels):
            return False

        for i in range(n_seen):
            if np.all(m[:, i] == m[0, i]):
                return False

        for i, v in enumerate(m):
            if np.all(v == EXCLUSIVE):
                # uncovered node
                return False
            else:
                for j in range(i + 1, len(unseen_labels)):
                    if np.all(v == m[j]):
                        return False
        return True

        # # for invalid
        # for i, v in enumerate(m):
        #     if np.all(v == EXCLUSIVE):
        #         # uncovered node
        #         return False
        #     else:
        #         for j in range(i + 1, len(unseen_labels)):
        #             if np.all(v == m[j]):
        #                 return True
        # return False

    def _check_exist_equal(self, nodes):
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if self.get_relation(nodes[i], nodes[j]) == EQUAL:
                    return True
        return False

    def _sample_non_exclusive_neighbors(self, nodes, n_neighbor, candidates=None, one_hop=False, tol=1000):
        neighbors = set()
        if candidates is None:
            candidates = self.candidate_labels
        if one_hop:
            for l in nodes:
                neighbors.update([n for n in self.graph_undirected_view.neighbors(l) if n in candidates and self.get_relation(l, n) != EXCLUSIVE])
        else:
            for n in candidates:
                for l in nodes:
                    r = self.get_relation(l, n)
                    if r != EXCLUSIVE and r != EQUAL:
                        neighbors.add(n)
                        break

        invalid = set(nodes)
        neighbors = [n for n in neighbors if n not in invalid]

        if len(neighbors) < n_neighbor:
            return []

        cnt = 0
        while cnt < tol:
            cnt += 1
            sampled_neighbors = self.generator.choice(neighbors, n_neighbor, replace=False)
            if self._check_unseen_valid(nodes, sampled_neighbors):
                return sampled_neighbors
        return []

    @staticmethod
    def _is_subclique(G, nodelist):
        H = G.subgraph(nodelist)
        n = len(nodelist)
        return H.size() == n * (n - 1) / 2

    def _sample_exclusive_connected_subgraph(self, nodes, n_node, n_subgraph, force_cover=False, tol=1000):
        n = len(nodes)
        if isinstance(n_node, list):
            assert n >= max(n_node)
        else:
            assert n >= n_node
            n_node = [n_node]

        # assume n_node > 1
        graph = self.exclusive_graph.subgraph(nodes)

        if force_cover:
            if len(graph.nodes) < n:
                return []
            min_n_neighbor = min(n_node)
            for node in graph.nodes:
                if graph.degree(node) < min_n_neighbor:
                    return []

        n_node_to_candidates = {ni: [i for i in graph.nodes if graph.degree(i) >= (ni - 1)] for ni in n_node}
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

    def _sample_label_graph(self,
                            candidates,
                            n_desired_labels,
                            n_all_labels,
                            tol=1000,
                            ):

        # sample exclusive desired labels, which should be non-isolated
        desired_labels = self._sample_exclusive_connected_subgraph(
            candidates,
            n_desired_labels,
            1,
            force_cover=False,
            tol=tol
        )
        if len(desired_labels) == 0:
            return None
        desired_labels = desired_labels[0]

        seen_labels = self._sample_non_exclusive_neighbors(
            desired_labels,
            n_all_labels - n_desired_labels,
            candidates,
            tol=tol
        )
        if len(seen_labels) == 0:
            return None

        all_labels = list(seen_labels) + list(desired_labels)
        assert len(all_labels) == n_all_labels
        if self._check_exist_equal(all_labels):
            return None

        label_relations = []
        for i in range(n_all_labels):
            li = all_labels[i]
            for j in range(i + 1, n_all_labels):
                lj = all_labels[j]
                r = self.get_relation(li, lj)
                label_relations.append((li, lj, r))

        return label_relations, desired_labels, all_labels

    def _sample(self,
                candidates,
                n_desired_labels,
                n_all_labels,
                n_models,
                model_label_space_lo,
                model_label_space_hi,
                tol=1000
                ):

        res = self._sample_label_graph(
            candidates,
            n_desired_labels,
            n_all_labels,
            tol=tol
        )
        if res is None:
            return None

        label_relations, desired_labels, all_labels = res
        seen_labels = [i for i in all_labels if i not in desired_labels]

        # sample exclusive label spaces for models
        # make sure all the seen labels are covered
        all_model_labels = self._sample_exclusive_connected_subgraph(
            list(seen_labels),
            list(range(model_label_space_lo, model_label_space_hi + 1)),
            n_models,
            force_cover=True,
            tol=tol
        )
        if len(all_model_labels) == 0:
            return None

        return label_relations, all_model_labels, desired_labels, all_labels

    def sample_dataset(self,
                       n_samples,
                       n_desired_labels,
                       n_all_labels,
                       n_models,
                       model_label_space_lo,
                       model_label_space_hi):

        sampled_label_relations = []
        sampled_model_labels = []
        sampled_desired_labels = []
        sampled_all_labels = []

        for _ in trange(n_samples):
            while True:
                node = self.generator.choice(self.candidate_labels)
                candidates = set(nx.descendants(self.graph, node)) & set(self.candidate_labels)
                if 2 * n_all_labels < len(candidates) < 4 * n_all_labels:
                    results = self._sample(
                        list(candidates),
                        n_desired_labels,
                        n_all_labels,
                        n_models,
                        model_label_space_lo,
                        model_label_space_hi
                    )
                    if results is not None:
                        label_relations, all_model_labels, desired_labels, all_labels = results
                        break

            sampled_label_relations.append(label_relations)
            sampled_model_labels.append(all_model_labels)
            sampled_desired_labels.append(desired_labels)
            sampled_all_labels.append(all_labels)

        return sampled_label_relations, sampled_model_labels, sampled_desired_labels, sampled_all_labels

    def sample_label_graph(self,
                           n_samples,
                           n_desired_labels,
                           n_all_labels):
        assert n_desired_labels < n_all_labels

        sampled_label_relations = []
        sampled_desired_labels = []
        sampled_all_labels = []

        for _ in trange(n_samples):
            while True:
                node = self.generator.choice(self.candidate_labels)
                candidates = set(nx.descendants(self.graph, node)) & set(self.candidate_labels)
                if 1 * n_all_labels < len(candidates) < 3 * n_all_labels:
                    results = self._sample_label_graph(
                        list(candidates),
                        n_desired_labels,
                        n_all_labels,
                    )
                    if results is not None:
                        label_relations, desired_labels, all_labels = results
                        break

            sampled_label_relations.append(label_relations)
            sampled_desired_labels.append(desired_labels)
            sampled_all_labels.append(all_labels)

        return sampled_label_relations, sampled_desired_labels, sampled_all_labels


class LSHTCDataset(HierDataset):

    def __init__(self, path, label_min_support=500, random_state=12345, preprocessed_path=None):
        dataset = pickle.load(open(path, "rb"))
        super(LSHTCDataset, self).__init__(
            dataset=dataset,
            label_min_support=label_min_support,
            random_state=random_state,
            preprocessed_path=preprocessed_path
        )
        self.features = dataset['embeds']
        self.raw_data = dataset['raw_text']

    def get_dataset(self, classes_name, sample_number=-1, disjoint=True):
        class_to_data_idx, instance_idx = self._get_classes_data(
            classes_name=classes_name, sample_number=sample_number, disjoint=disjoint)

        instance_to_label = reverse_dict(class_to_data_idx)
        y = np.array([instance_to_label[i] for i in instance_idx])
        X = self.features[instance_idx]

        return TextDataset(X=X, y=y, raw_data=[self.raw_data[i] for i in instance_idx], classes=classes_name)

    def get_binary_dataset(self, class_name, sample_number=-1):
        class_to_data_idx, instance_idx = self._get_binary_data(
            class_name=class_name, sample_number=sample_number)

        instance_to_label = reverse_dict(class_to_data_idx)
        y = np.array([instance_to_label[i] for i in instance_idx])
        X = self.features[instance_idx]

        return TextDataset(X=X, y=y, raw_data=[self.raw_data[i] for i in instance_idx], classes=[ABSTAIN, class_name])


class ImageNetDataset(HierDataset):
    """
    This is the Hierarchy Image Dataset for ImageNet with WordNet taxonomy
    Path config:
        root:
            words.txt
            wordnet.is_a.txt
            train:
                class_name:
                    file_name
    """

    def __init__(self, path, label_min_support=500, random_state=12345, preprocessed_path=None):
        dataset = pickle.load(open(path, "rb"))
        super(ImageNetDataset, self).__init__(
            dataset=dataset,
            label_min_support=label_min_support,
            random_state=random_state,
            preprocessed_path=preprocessed_path
        )
        self.raw_data = dataset['raw_image']
        self.name2label = dataset['name2label']
        self.label2name = dataset['label2name']
        self.image2path = dataset['image2path']

    def get_dataset(self, classes_name, sample_number=-1, disjoint=True):
        class_to_data_idx, instance_idx = self._get_classes_data(
            classes_name=classes_name, sample_number=sample_number, disjoint=disjoint)

        instance_to_label = reverse_dict(class_to_data_idx)
        y = np.array([instance_to_label[i] for i in instance_idx])

        return ImageDataset(y=y, raw_data=[self.raw_data[i] for i in instance_idx], image2path=self.image2path, classes=classes_name)

    def get_binary_dataset(self, class_name, sample_number=-1):
        class_to_data_idx, instance_idx = self._get_binary_data(
            class_name=class_name, sample_number=sample_number)

        instance_to_label = reverse_dict(class_to_data_idx)
        y = np.array([instance_to_label[i] for i in instance_idx])

        return ImageDataset(y=y, raw_data=[self.raw_data[i] for i in instance_idx], image2path=self.image2path, classes=[ABSTAIN, class_name])


class BaseDataset:
    y: np.ndarray

    def __len__(self):
        return len(self.y)

    def split(self, split=(0.8, 0.1, 0.1)):
        n = len(self)
        perm_idx = np.random.permutation(n)
        split_point1 = int(n * split[0])
        split_point2 = int(n * (split[0] + split[1]))

        train_idx = perm_idx[:split_point1]
        train_data = self.get_subset(train_idx)

        valid_idx = perm_idx[split_point1:split_point2]
        valid_data = self.get_subset(valid_idx)

        test_idx = perm_idx[split_point2:]
        test_data = self.get_subset(test_idx)
        return train_data, valid_data, test_data

    def sample(self, n: int):
        if n >= len(self):
            return self
        idx = np.random.choice(len(self), n, replace=False)
        return self.get_subset(idx)

    @abstractmethod
    def get_subset(self, idx):
        pass


class TextDataset(BaseDataset):
    def __init__(self, X, y, raw_data, classes, convert_y=True):
        self.classes = classes
        self.class_to_id = {c: i for i, c in enumerate(classes)}
        self.X = X
        if convert_y:
            self.y = np.array([self.class_to_id[i] for i in y])
        else:
            self.y = y
        self.raw_data = raw_data

    def get_subset(self, idx):
        return TextDataset(
            X=self.X[idx],
            y=self.y[idx],
            raw_data=[self.raw_data[i] for i in idx],
            classes=self.classes,
            convert_y=False
        )


class ImageDataset(BaseDataset):
    def __init__(self, y, raw_data, classes, image2path, convert_y=True):
        self.classes = classes
        self.class_to_id = {c: i for i, c in enumerate(classes)}
        if convert_y:
            self.y = np.array([self.class_to_id[i] for i in y])
        else:
            self.y = y
        self.raw_data = raw_data
        self.image2path = image2path

    def get_subset(self, idx):
        raw_data = [self.raw_data[i] for i in idx]
        image2path = self.image2path
        return ImageDataset(
            y=self.y[idx],
            raw_data=raw_data,
            classes=self.classes,
            image2path={k: image2path[k] for k in raw_data},
            convert_y=False
        )
