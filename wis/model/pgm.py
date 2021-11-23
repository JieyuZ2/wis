import logging
from abc import ABC
from collections import defaultdict
from itertools import chain, permutations
from typing import Dict, List

import numpy as np
import torch
from numba import jit
from numba.typed import List as nbList
from sklearn.metrics import accuracy_score, f1_score
from tqdm import trange

from ..label import OVERLAP, INCLUSION, INCLUDED, EXCLUSIVE, ABSTAIN
from ..mytype import Config
from ..utils import cartesian, probs_to_preds

logger = logging.getLogger(__name__)

USE_RELATIONS = [INCLUDED, INCLUSION, EXCLUSIVE, OVERLAP]

ACC_FACTOR = -1
PSEUDO_ACC_FACTOR = -2
ABSTAIN_ACC_FACTOR = -3
ABSTAIN_PSEUDO_ACC_FACTOR = -4


class TrainConfig(Config):
    gpu: bool = False
    exact: bool = False

    n_epochs: int = 10
    project: bool = False
    step_size: float = -1
    # step_size: float = 0.1
    decay: float = -1
    reg_weight: float = 0.0
    alpha: int = 1.0
    interval: int = 1
    print_interval: int = 100
    patience: int = 10000

    # n_iter: int = 100
    # burnin: int = 10
    # aver_every: int = 2

    n_iter: int = 1
    burnin: int = 0
    aver_every: int = 1


def check_label_swapping(proba, y):
    n, k = proba.shape
    best_f1 = -1
    best_acc = -1
    for p in permutations(range(k)):
        proba_p = proba[:, p]
        y_pred = probs_to_preds(proba_p)
        f1_macro = f1_score(y, y_pred, average='macro')
        acc = accuracy_score(y, y_pred)
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_acc = acc
            best_perm = p
            best_proba = proba_p
    return best_proba, best_perm, best_f1, best_acc


class AbstractPGM(ABC):
    gpu: bool

    n_cartesian_y: int
    model_prod_cardinalities: np.ndarray
    cartesian: np.ndarray
    ss: np.ndarray

    nodes: Dict
    n_nodes: int
    factors: List
    n_factors: int

    s_factors: nbList

    cardinalities: np.ndarray
    all_node_ids: np.ndarray
    y_node_ids: np.ndarray
    model_node_ids: np.ndarray

    assignment: np.ndarray
    ess: np.ndarray
    Z: np.ndarray

    theta: np.ndarray

    def __init__(self, n_nodes, n_models, n_factors, cardinalities, exact, gpu):
        exact = exact or gpu
        self.gpu = gpu
        self.exact = exact
        self.exact_inference = (n_nodes - n_models) < 20
        logger.info(f'exact inference: {self.exact_inference} # of nodes: {(n_nodes - n_models)}')
        self.n_nodes = n_nodes
        self.n_factors = n_factors
        self.cardinalities = np.array(cardinalities, dtype=int)

        self.relation_theta_idx = np.array([i for i, (_, _, v, _) in enumerate(self.factors) if v != -1], dtype=int)

        self.s_factors = nbList()
        [self.s_factors.append(i) for i in self.factors]

        self.model_node_ids = np.arange(n_nodes - n_models, n_nodes)
        self.y_node_ids = np.arange(0, n_nodes - n_models)
        self.all_node_ids = np.arange(0, n_nodes)

        model_cardinalities = self.cardinalities[-n_models:]
        self.model_prod_cardinalities = np.empty(n_models, dtype=int)
        self.model_prod_cardinalities[-1] = 1
        for i in reversed(range(n_models - 1)):
            self.model_prod_cardinalities[i] = model_cardinalities[i + 1] * self.model_prod_cardinalities[i + 1]

        self.cartesian_y = cartesian(self.cardinalities[:-n_models]).astype(np.int8)
        if exact:
            model_cartesian = cartesian(model_cardinalities).astype(np.int8)
            self.cartesian_y = cartesian(self.cardinalities[:-n_models]).astype(np.int8)
            self.cartesian = np.zeros((np.product(self.cardinalities), self.n_nodes), dtype=np.int8)
            self.n_cartesian_y = len(self.cartesian_y)
            for i, model_assignment in enumerate(model_cartesian):
                self.cartesian[self.n_cartesian_y * i:self.n_cartesian_y * (i + 1), self.model_node_ids] = model_assignment
                self.cartesian[self.n_cartesian_y * i:self.n_cartesian_y * (i + 1), self.y_node_ids] = self.cartesian_y

            if gpu:
                self.relation_theta_idx = torch.LongTensor(self.relation_theta_idx).cuda()
                ss = self.calc_sufficient_statistic(self.cartesian)
                self.ss = torch.tensor(ss, dtype=torch.float32, requires_grad=False).cuda()
            else:
                self.ss = self.calc_sufficient_statistic(self.cartesian).astype(np.int8)  # np.array([self.calc_sufficient_statistic(i) for i in self.cartesian], dtype=int)
        else:
            max_card = max(self.cardinalities)
            self.assignment = self.random_state()
            self.ess = np.zeros((max_card, self.n_factors), dtype=int)
            self.ess[:] = self.calc_sufficient_statistic(self.assignment.reshape(1, -1))
            self.Z = np.zeros(max_card)
            self.ss_cache = {}

        self.theta = np.ones(self.n_factors)

    def _init_desired_mappings(self, desired_classes, label_relations):
        self.desired_mappings = defaultdict(list)
        for m, n, r in label_relations:
            # if (r != EXCLUSIVE) and (r in USE_RELATIONS):
            if (r != EXCLUSIVE):
                if n in desired_classes and m not in desired_classes:
                    self.desired_mappings[m].append(n)
                elif m in desired_classes and n not in desired_classes:
                    self.desired_mappings[n].append(m)

    def _init_label_node(self, desired_classes, extra_class=None):
        cardinalities = [len(desired_classes)]
        self.nodes = {0: []}
        self.label_to_node = {l: 0 for l in desired_classes}
        node_id = 1
        if extra_class is not None:
            for l in extra_class:
                if l != ABSTAIN:
                    self.nodes[node_id] = []
                    self.label_to_node[l] = node_id
                    cardinalities.append(2)
                    node_id += 1
        return cardinalities, node_id

    def _init_label_relation_factor(self, factor_id, desired_classes, label_relations):
        for i, (y1, y2, relation_type) in enumerate(label_relations):
            if relation_type in USE_RELATIONS:
                y1_node, y2_node = self.label_to_node[y1], self.label_to_node[y2]
                if y1_node != y2_node:
                    if y1_node == 0:
                        self.factors.append((y1_node, y2_node, relation_type, (desired_classes.index(y1), 0)))
                    elif y2_node == 0:
                        self.factors.append((y1_node, y2_node, relation_type, (desired_classes.index(y2), 1)))
                    else:
                        self.factors.append((y1_node, y2_node, relation_type, (-1, -1)))
                    self.nodes[y1_node].append((factor_id, 0))
                    self.nodes[y2_node].append((factor_id, 1))
                    factor_id += 1
        return factor_id

    @staticmethod
    def _convert_L(L):
        return np.where(L == -1, 0, L)

    def random_state(self):
        return np.array([np.random.randint(card) for card in self.cardinalities], dtype=int)

    def calc_sufficient_statistic(self, assignments):
        return calc_sufficient_statistic(self.s_factors, assignments)

    def exact_one_step_gpu(self, Ls, theta, **kwargs):
        idx = Ls @ self.model_prod_cardinalities
        start, end = idx * self.n_cartesian_y, (idx + 1) * self.n_cartesian_y
        p = torch.exp(self.ss @ theta).view(-1, 1)
        t = p * self.ss
        ss_a = t.sum(0) / p.sum()
        ss_b = torch.sum(t[start:end], 0) / torch.sum(p[start:end])
        return ss_a, ss_b

    def exact_one_step(self, Ls, **kwargs):
        idx = Ls @ self.model_prod_cardinalities
        start, end = idx * self.n_cartesian_y, (idx + 1) * self.n_cartesian_y
        p = np.exp(self.ss @ self.theta.reshape(-1, 1))
        t = p * self.ss
        ss_a = t.sum(0) / p.sum()
        ss_b = np.sum(t[start:end], 0) / np.sum(p[start:end])
        return ss_a, ss_b

    def gibbs_sampling_assignment(self, nodes, n_iter=100, burnin=10, aver_every=2):
        return gibbs_sampling_assignment(self.Z, self.assignment, self.cardinalities, self.s_factors, self.theta, nodes, n_iter=n_iter, burnin=burnin, aver_every=aver_every)

    def gibbs_sampling_statistics(self, nodes, n_iter=100, burnin=10, aver_every=2):
        return gibbs_sampling_statistics(self.ess, self.Z, self.assignment, self.cardinalities, self.s_factors, self.theta, nodes, n_iter=n_iter, burnin=burnin, aver_every=aver_every)

    def approximate_one_step(self, Ls, n_iter=100, burnin=10, aver_every=2, **kwargs):
        self.assignment[self.model_node_ids] = Ls
        ss_b = self.gibbs_sampling_statistics(self.y_node_ids, n_iter=n_iter, burnin=burnin, aver_every=aver_every)
        ss_a = self.gibbs_sampling_statistics(self.all_node_ids, n_iter=n_iter, burnin=burnin, aver_every=aver_every)
        return ss_a, ss_b

    def infer(self, Ls_list, exact=True, **kwargs):
        Ls_list = self._convert_L(Ls_list)
        res = np.zeros((len(Ls_list), self.n_desired_classes))
        cache = {}
        if exact:
            if self.gpu:
                theta = torch.tensor(self.theta, dtype=torch.float32).cuda()
                for i, Ls in enumerate(Ls_list):
                    idx = Ls @ self.model_prod_cardinalities
                    if idx in cache:
                        res[i] = cache[idx]
                    else:
                        ss = self.ss[idx * self.n_cartesian_y: (idx + 1) * self.n_cartesian_y]
                        potentials = torch.exp(ss @ theta).view(self.n_desired_classes, -1).sum(1)
                        prob = (potentials / potentials.sum()).cpu().numpy()
                        res[i] = prob
                        cache[idx] = prob
            else:
                for i, Ls in enumerate(Ls_list):
                    idx = Ls @ self.model_prod_cardinalities
                    if idx in cache:
                        res[i] = cache[idx]
                    else:
                        ss = self.ss[idx * self.n_cartesian_y: (idx + 1) * self.n_cartesian_y]
                        potentials = np.exp(ss @ self.theta).reshape(self.n_desired_classes, -1).sum(1)
                        prob = potentials / potentials.sum()
                        res[i] = prob
                        cache[idx] = prob
        else:
            if self.exact_inference:
                assignments = np.zeros((len(self.cartesian_y), self.n_nodes))
                assignments[:, self.y_node_ids] = self.cartesian_y
                for i, Ls in enumerate(Ls_list):

                    idx = Ls @ self.model_prod_cardinalities
                    if idx in cache:
                        res[i] = cache[idx]
                    else:
                        if idx in self.ss_cache:
                            ss = self.ss_cache[idx]
                        else:
                            assignments[:, self.model_node_ids] = Ls
                            ss = self.calc_sufficient_statistic(assignments)
                            self.ss_cache[idx] = ss
                        potentials = np.exp(ss @ self.theta).reshape(self.n_desired_classes, -1).sum(1)
                        prob = potentials / potentials.sum()

                        res[i] = prob
                        cache[idx] = prob
            else:
                for i, Ls in enumerate(Ls_list):
                    idx = Ls @ self.model_prod_cardinalities
                    if idx in cache:
                        value, prob = cache[idx]
                    else:
                        self.assignment[self.model_node_ids] = Ls
                        assign_chain = self.gibbs_sampling_assignment(self.y_node_ids, n_iter=10000, burnin=100, aver_every=2)
                        assignment = assign_chain[:, 0]
                        value, counts = np.unique(assignment, return_counts=True)
                        prob = counts / counts.sum()
                        cache[idx] = (value, prob)
                    res[i][value] = prob
        if hasattr(self, 'label_perm'):
            res = res[:, self.label_perm]
        return res

    def marginal_likelihood(self, Ls_list, exact=True):
        Ls_list = self._convert_L(Ls_list)
        log_prob = 0
        cache = {}
        Ls_list = np.array(Ls_list)
        if exact:
            if self.gpu:
                theta = torch.tensor(self.theta, dtype=torch.float32).cuda()
                potentials = torch.exp(self.ss @ theta)
                logZ = torch.log(potentials.sum())
                for Ls in Ls_list:
                    idx = Ls @ self.model_prod_cardinalities
                    if idx in cache:
                        log_prob += cache[idx]
                    else:
                        potential = potentials[idx * self.n_cartesian_y: (idx + 1) * self.n_cartesian_y]
                        log_p = torch.log(potential.sum()).item()
                        log_prob += log_p
                        cache[idx] = log_p
            else:
                theta = self.theta
                potentials = np.exp(self.ss @ theta)
                logZ = np.log(potentials.sum())
                for Ls in Ls_list:
                    idx = Ls @ self.model_prod_cardinalities
                    if idx in cache:
                        log_prob += cache[idx]
                    else:
                        potential = potentials[idx * self.n_cartesian_y: (idx + 1) * self.n_cartesian_y]
                        log_p = np.log(potential.sum())
                        log_prob += log_p
                        cache[idx] = log_p
            log_prob = - (log_prob / len(Ls_list) - logZ)
        else:
            gibbs_chain = self.gibbs_sampling_assignment(self.all_node_ids, n_iter=100000, burnin=100, aver_every=2)
            gibbs_chain = gibbs_chain[:, self.model_node_ids]
            total_n = len(gibbs_chain)
            for Ls in Ls_list:
                idx = Ls @ self.model_prod_cardinalities
                if idx in cache:
                    log_prob += cache[idx]
                else:
                    cnt = np.all(gibbs_chain == Ls, axis=1)
                    log_p = np.log((cnt.sum() + 1) / total_n)
                    log_prob += log_p
                    cache[idx] = log_p
            log_prob = - log_prob / len(Ls_list)
        return log_prob

    def fit(self,
            Ls_list,
            valid_data=None,
            init_y=None,
            init_theta=None,
            exact=False,
            project=False,
            n_epochs=100,
            step_size=-1,
            decay=-1,
            reg_weight=0.0,
            alpha=0.0,
            interval=1,
            print_interval=1000,
            patience=10,
            **kwargs
            ):

        Ls_list = self._convert_L(Ls_list)
        if hasattr(self, 'init_theta') and (init_y is not None or init_theta is not None):
            self.theta = self.init_theta(Ls_list, init_y, init_theta)

        n = len(Ls_list)
        if step_size == -1:
            step_size = 1 / n
        if decay == -1:
            decay = 0.001 ** (1 / n_epochs)

        exact = exact or self.gpu
        logger.info(f'start training {self.__class__.__name__}: gpu={self.gpu} exact={exact} project={project} step_size={step_size} decay={decay} reg_weight={reg_weight} alpha={alpha}')

        if self.gpu:
            theta = torch.tensor(self.theta, dtype=torch.float32).cuda()
        else:
            theta = self.theta

        valid_flag = False
        if valid_data is not None:
            valid_flag = True
            Ls_list_valid, y_valid = valid_data

            res = self.infer(Ls_list_valid, exact=exact)
            y_pred = probs_to_preds(res)
            # acc = accuracy_score(y_valid, y_pred)
            # f1_macro = f1_score(y_valid, y_pred, average='macro')

            perm_proba, label_perm, perm_f1, perm_acc = check_label_swapping(res, y_valid)

            logger.info(f'init:\tacc: {perm_acc:.4f}\tf1: {perm_f1:.4f}')

            best_valid_f1 = perm_f1
            best_label_perm = label_perm
            best_epoch = 0

            not_improve_cnt = 0
            if self.gpu:
                best_model = theta.cpu().numpy()
            else:
                best_model = theta.copy()

        cnt = 0
        early_stop = False
        for i in trange(n_epochs, disable=valid_flag):
            perm = np.random.permutation(n)
            for Ls in Ls_list[perm]:
                cnt += 1
                if self.gpu:
                    ss_a, ss_b = self.exact_one_step_gpu(Ls, theta, **kwargs)
                else:
                    if exact:
                        ss_a, ss_b = self.exact_one_step(Ls, **kwargs)
                    else:
                        ss_a, ss_b = self.approximate_one_step(Ls, **kwargs)

                diff = (ss_a - ss_b) + 2 * reg_weight * (theta - alpha)
                theta -= step_size * diff
                # if isinstance(self, PLRM) and init_theta is not None:
                #     theta[self.pseudo_acc_idx] = init_theta

                if project:
                    if self.gpu:
                        # theta[self.relation_theta_idx] = (torch.abs(theta[self.relation_theta_idx]) + theta[self.relation_theta_idx]) / 2
                        theta = (torch.abs(theta) + theta) / 2
                    elif self.relation_theta_idx.size != 0:
                        # theta[self.relation_theta_idx] = (np.abs(theta[self.relation_theta_idx]) + theta[self.relation_theta_idx]) / 2
                        theta = (np.abs(theta) + theta) / 2

                if valid_flag:
                    if cnt % interval == 0:
                        if self.gpu:
                            self.theta = theta.cpu().numpy()
                        res = self.infer(Ls_list_valid, exact=exact)
                        y_pred = probs_to_preds(res)

                        # acc = accuracy_score(y_valid, y_pred)
                        # f1_macro = f1_score(y_valid, y_pred, average='macro')
                        perm_proba, label_perm, perm_f1, perm_acc = check_label_swapping(res, y_valid)

                        if cnt % print_interval == 0:
                            logger.info(f'epoch@{i}/{n_epochs}:\tacc: {perm_acc * 100:.2f}\tf1: {perm_f1 * 100:.2f}'
                                        f'\tbest_f1: {best_valid_f1 * 100:.2f} @ epoch {best_epoch}')

                        # mar = self.marginal_likelihood(Ls_list, exact=exact)
                        # logger.info(f'epoch@{i}/{n_epochs}:\tacc: {acc:.4f}\tf1_macro: {f1_macro:.4f}\tlld: {mar:.4f}\tperm_f1: {perm_f1:.4f}')

                        if perm_f1 > best_valid_f1:
                            best_valid_f1 = perm_f1
                            not_improve_cnt = 0
                            best_label_perm = label_perm
                            best_epoch = i
                            if self.gpu:
                                best_model = theta.cpu().numpy()
                            else:
                                best_model = theta.copy()
                        else:
                            not_improve_cnt += 1
                            if patience > 0 and not_improve_cnt >= patience:
                                early_stop = True
                                break

            if early_stop:
                logger.info(f'early stop @ epoch {i}!')
                break
            step_size *= decay

        if valid_flag:
            self.theta = best_model
            self.label_perm = best_label_perm
        else:
            if self.gpu:
                self.theta = theta.cpu().numpy()


class PGM(AbstractPGM):
    def __init__(self, desired_classes, model_classes, exact, gpu=False):
        self.desired_classes = desired_classes
        self.model_classes = model_classes
        self.all_classes = sorted(set(chain.from_iterable(model_classes)).union(desired_classes))
        self.n_desired_classes = len(desired_classes)
        self.n_models = len(model_classes)
        self.n_all_classes = len(self.all_classes)

        cardinalities, node_id = self._init_label_node(desired_classes)

        self.factors = []
        factor_id = 0
        for model_id, model_classes in enumerate(self.model_classes):
            cardinalities.append(len(model_classes))
            self.nodes[node_id] = []
            for label_id, l in enumerate(model_classes):
                if l in self.desired_classes:
                    l_node = self.label_to_node[l]
                    self.nodes[node_id].append((factor_id, 1))
                    self.nodes[l_node].append((factor_id, 0))
                    self.factors.append((l_node, node_id, ACC_FACTOR, (self.desired_classes.index(l), label_id)))
                    factor_id += 1
            node_id += 1

        super(PGM, self).__init__(node_id, self.n_models, factor_id, cardinalities, exact, gpu)


class LFPGM(AbstractPGM):
    # use labeling function only
    def __init__(self,
                 desired_classes,
                 model_classes,
                 label_relations,
                 exact,
                 gpu=False,
                 **kwargs):
        self.desired_classes = desired_classes
        self.model_classes = model_classes
        self.all_classes = sorted(set(chain.from_iterable(model_classes)).union(desired_classes))
        self.n_desired_classes = len(desired_classes)
        self.n_models = len(model_classes)
        self.n_all_classes = len(self.all_classes)

        self._init_desired_mappings(desired_classes, label_relations)
        self.label_relations = label_relations

        cardinalities, node_id = self._init_label_node(desired_classes)

        self.factors = []
        factor_id = 0
        # init acc (desired) / pseudo acc factor
        for model_id, model_classes in enumerate(self.model_classes):
            cardinalities.append(len(model_classes))
            self.nodes[node_id] = []
            acc_factor = ABSTAIN_ACC_FACTOR if ABSTAIN in model_classes else ACC_FACTOR
            pseudo_acc_factor = ABSTAIN_PSEUDO_ACC_FACTOR if ABSTAIN in model_classes else PSEUDO_ACC_FACTOR
            for label_id, l in enumerate(model_classes):
                if l in self.desired_classes:
                    l_node = self.label_to_node[l]
                    self.nodes[node_id].append((factor_id, 1))
                    self.nodes[l_node].append((factor_id, 0))
                    self.factors.append((l_node, node_id, acc_factor, (self.desired_classes.index(l), label_id)))
                    factor_id += 1
                elif l in self.desired_mappings:
                    ns = self.desired_mappings[l]
                    nodes = []
                    for n in ns:
                        l_node = self.label_to_node[n]
                        nodes.append((factor_id, 1))
                        self.nodes[l_node].append((factor_id, 0))
                        self.factors.append((l_node, node_id, pseudo_acc_factor, (self.desired_classes.index(n), label_id)))
                        factor_id += 1
                    self.nodes[node_id].extend(nodes)
            node_id += 1

        super(LFPGM, self).__init__(node_id, self.n_models, factor_id, cardinalities, exact, gpu)

    def init_theta(self, Ls_list, init_y, init_theta):
        assignments = np.hstack((init_y.reshape(-1, 1), Ls_list))
        ss = self.calc_sufficient_statistic(assignments)
        ess = np.mean(ss, 0)
        return ess


class PLRM(AbstractPGM):
    def __init__(self,
                 desired_classes,
                 model_classes,
                 label_relations,
                 exact,
                 gpu=False,
                 **kwargs
                 ):
        self.desired_classes = desired_classes
        self.model_classes = model_classes
        self.all_classes = sorted(set(chain.from_iterable(model_classes)).union(desired_classes))
        self.extra_class = [i for i in self.all_classes if i not in self.desired_classes]
        self.n_desired_classes = len(desired_classes)
        self.n_models = len(model_classes)
        self.n_all_classes = len(self.all_classes)

        self.label_relations = label_relations
        self._init_desired_mappings(desired_classes, label_relations)

        cardinalities, node_id = self._init_label_node(desired_classes, extra_class=self.extra_class)

        self.factors = []
        factor_id = self._init_label_relation_factor(0, desired_classes, label_relations)

        pseudo_acc_idx = []
        acc_idx = []

        # init acc (desired / undesired) / pseudo acc factor
        for model_id, model_classes in enumerate(self.model_classes):
            cardinalities.append(len(model_classes))
            self.nodes[node_id] = []
            acc_factor = ABSTAIN_ACC_FACTOR if ABSTAIN in model_classes else ACC_FACTOR
            pseudo_acc_factor = ABSTAIN_PSEUDO_ACC_FACTOR if ABSTAIN in model_classes else PSEUDO_ACC_FACTOR
            for label_id, l in enumerate(model_classes):
                if l != ABSTAIN:
                    l_node = self.label_to_node[l]
                    self.nodes[node_id].append((factor_id, 1))
                    self.nodes[l_node].append((factor_id, 0))
                    if l_node == 0:
                        # desired acc factor
                        self.factors.append((l_node, node_id, acc_factor, (self.desired_classes.index(l), label_id)))
                        acc_idx.append(factor_id)
                        factor_id += 1
                    else:
                        # undesired acc factor
                        self.factors.append((l_node, node_id, acc_factor, (-1, label_id)))
                        acc_idx.append(factor_id)
                        factor_id += 1
                        ns = self.desired_mappings[l]
                        nodes = []
                        for n in ns:
                            l_node = self.label_to_node[n]
                            nodes.append((factor_id, 1))
                            self.nodes[l_node].append((factor_id, 0))
                            # pseudo acc factor
                            pseudo_acc_idx.append(factor_id)
                            self.factors.append((l_node, node_id, pseudo_acc_factor, (self.desired_classes.index(n), label_id)))
                            factor_id += 1
                        self.nodes[node_id].extend(nodes)
            node_id += 1

        self.pseudo_acc_idx = pseudo_acc_idx
        self.acc_idx = acc_idx

        super(PLRM, self).__init__(node_id, self.n_models, factor_id, cardinalities, exact, gpu)

    def init_theta(self, Ls_list, init_y, init_theta):
        theta = np.zeros(self.n_factors)
        theta[self.acc_idx] = 1.0
        theta[self.pseudo_acc_idx] = init_theta
        return theta


@jit(nopython=True, cache=True, nogil=True)
def eval_factor(f, y1, y2, para1, para2):
    if f == ACC_FACTOR:
        return acc_phi(y1, y2, para1, para2)
    elif f == PSEUDO_ACC_FACTOR:
        return pseudo_acc_phi(y1, y2, para1, para2)
    elif f == ABSTAIN_ACC_FACTOR:
        return abstain_acc_phi(y1, y2, para1, para2)
    elif f == ABSTAIN_PSEUDO_ACC_FACTOR:
        return abstain_pseudo_acc_phi(y1, y2, para1, para2)
    elif f == EXCLUSIVE:
        return exclusion_phi(y1, y2, para1, para2)
    elif f == OVERLAP:
        return overlap_phi(y1, y2, para1, para2)
    elif f == INCLUSION:
        return subsuming_phi(y1, y2, para1, para2)
    elif f == INCLUDED:
        return subsumed_phi(y1, y2, para1, para2)


@jit(nopython=True, nogil=True)
def acc_phi(y, l, y_id, l_id):
        if y_id != -1:
            y = y == y_id
        l = l == l_id
        return y == l

    # if l == l_id:
    #     if y_id != -1:
    #         y = y == y_id
    #     if y == 1:
    #         return 1
    #     else:
    #         return -1
    # return 0


@jit(nopython=True, nogil=True)
def abstain_acc_phi(y, l, y_id, l_id):
    if l == 0:
        return 0
    return acc_phi(y, l, y_id, l_id)


@jit(nopython=True, nogil=True)
def pseudo_acc_phi(y, l, y_id, l_id):
    if y_id != -1:
        y = y == y_id
    l = l == l_id
    return y == l

    # if l == l_id:
    #     if y_id != -1:
    #         y = y == y_id
    #     if y == 1:
    #         return 1
    #     else:
    #         return -1
    # return 0


@jit(nopython=True, nogil=True)
def abstain_pseudo_acc_phi(y, l, y_id, l_id):
    if l == 0:
        return 0
    return pseudo_acc_phi(y, l, y_id, l_id)


@jit(nopython=True, nogil=True)
def overlap_phi(y1, y2, label_id, pos):
    if pos == 1:
        y2 = y2 == label_id
    elif pos == 0:
        y1 = y1 == label_id
    return (y1 == y2 == 1)


@jit(nopython=True, nogil=True)
def exclusion_phi(y1, y2, label_id, pos):
    if pos == 1:
        y2 = y2 == label_id
    elif pos == 0:
        y1 = y1 == label_id
    if y1 == y2 == 1:
        return -1
    else:
        return 0


@jit(nopython=True, nogil=True)
def subsuming_phi(y1, y2, label_id, pos):
    if pos == 1:
        y2 = y2 == label_id
    elif pos == 0:
        y1 = y1 == label_id
    if y1 == 0 and y2 == 1:
        return -1
    else:
        return 0


@jit(nopython=True, nogil=True)
def subsumed_phi(y1, y2, label_id, pos):
    if pos == 1:
        y2 = y2 == label_id
    elif pos == 0:
        y1 = y1 == label_id
    if y1 == 1 and y2 == 0:
        return -1
    else:
        return 0


@jit(nopython=True, cache=True, nogil=True)
def calc_sufficient_statistic(s_factors, assignments):
    ss = np.zeros((len(assignments), len(s_factors)), dtype=np.int_)
    for j in range(len(assignments)):
        assignment = assignments[j]
        for i, (a, b, f, (para1, para2)) in enumerate(s_factors):
            ss[j][i] = eval_factor(f, assignment[a], assignment[b], para1, para2)
    return ss


@jit(parallel=False, nopython=True, cache=True, nogil=True)
def calc_sufficient_statistic_for_infer(s_factors, assignments, theta):
    sum_p = 0.0
    for j in range(len(assignments)):
        assignment = assignments[j]
        p = 0.0
        for i in range(len(s_factors)):
            a, b, f, (para1, para2) = s_factors[i]
            p += theta[i] * eval_factor(f, assignment[a], assignment[b], para1, para2)
        sum_p += np.exp(p)
    return sum_p


@jit(nopython=True, cache=True, nogil=True)
def gibbs_sampling_statistics(ss, Z, assignment, cardinalities, s_factors, theta, nodes, n_iter, burnin, aver_every):
    ess = np.zeros(len(s_factors))

    for i in range(1, burnin + 1):
        perm = np.random.permutation(len(nodes))
        for node in nodes[perm]:
            cardinality = cardinalities[node]
            for row_id, value in enumerate(range(cardinality)):
                p = 0.0
                for factor_id, (a, b, f, (para1, para2)) in enumerate(s_factors):
                    if a == node:
                        h = eval_factor(f, value, assignment[b], para1, para2)
                        p += h * theta[factor_id]
                    if b == node:
                        h = eval_factor(f, assignment[a], value, para1, para2)
                        p += h * theta[factor_id]
                Z[row_id] = np.exp(p)

            for j in range(1, cardinality):
                Z[j] += Z[j - 1]
            z = np.random.rand() * Z[cardinality - 1]
            chosen = np.argmax(Z[:cardinality] >= z)
            assignment[node] = chosen

    for i, (a, b, f, (para1, para2)) in enumerate(s_factors):
        ss[:, i] = eval_factor(f, assignment[a], assignment[b], para1, para2)

    chosen = 0
    cnt = 0
    for i in range(1, n_iter + 1):
        perm = np.random.permutation(len(nodes))
        for node in nodes[perm]:
            cardinality = cardinalities[node]
            for row_id, value in enumerate(range(cardinality)):
                p = 0.0
                for factor_id, (a, b, f, (para1, para2)) in enumerate(s_factors):
                    if a == node:
                        h = eval_factor(f, value, assignment[b], para1, para2)
                        ss[row_id, factor_id] = h
                        p += h * theta[factor_id]
                    if b == node:
                        h = eval_factor(f, assignment[a], value, para1, para2)
                        ss[row_id, factor_id] = h
                        p += h * theta[factor_id]
                Z[row_id] = np.exp(p)

            for j in range(1, cardinality):
                Z[j] += Z[j - 1]
            z = np.random.rand() * Z[cardinality - 1]
            chosen = np.argmax(Z[:cardinality] >= z)

            assignment[node] = chosen

            for j in range(cardinality):
                ss[j] = ss[chosen]

        if i % aver_every == 0:
            ess += ss[chosen]
            cnt += 1

    # return ess / ((n_iter - burnin) // aver_every)
    return ess / cnt


@jit(nopython=True, cache=True, nogil=True)
def gibbs_sampling_assignment(Z, assignment, cardinalities, s_factors, theta, nodes, n_iter, burnin, aver_every):
    n = ((n_iter - burnin) // aver_every)
    assign_chain = np.empty((n, len(cardinalities)), np.int_)
    cnt = 0

    for i in range(1, burnin + 1):
        perm = np.random.permutation(len(nodes))
        for node in nodes[perm]:
            cardinality = cardinalities[node]
            for row_id, value in enumerate(range(cardinality)):
                p = 0.0
                for factor_id, (a, b, f, (para1, para2)) in enumerate(s_factors):
                    if a == node:
                        h = eval_factor(f, value, assignment[b], para1, para2)
                        p += h * theta[factor_id]
                    if b == node:
                        h = eval_factor(f, assignment[a], value, para1, para2)
                        p += h * theta[factor_id]
                Z[row_id] = np.exp(p)

            for j in range(1, cardinality):
                Z[j] += Z[j - 1]
            z = np.random.rand() * Z[cardinality - 1]
            chosen = np.argmax(Z[:cardinality] >= z)
            assignment[node] = chosen

    for i in range(burnin + 1, n_iter + 1):
        perm = np.random.permutation(len(nodes))
        for node in nodes[perm]:
            cardinality = cardinalities[node]
            for row_id, value in enumerate(range(cardinality)):
                p = 0.0
                for factor_id, (a, b, f, (para1, para2)) in enumerate(s_factors):
                    if a == node:
                        h = eval_factor(f, value, assignment[b], para1, para2)
                        p += h * theta[factor_id]
                    if b == node:
                        h = eval_factor(f, assignment[a], value, para1, para2)
                        p += h * theta[factor_id]
                Z[row_id] = np.exp(p)

            for j in range(1, cardinality):
                Z[j] += Z[j - 1]
            z = np.random.rand() * Z[cardinality - 1]
            chosen = np.argmax(Z[:cardinality] >= z)

            assignment[node] = chosen

        if i % aver_every == 0:
            assign_chain[cnt] = assignment
            cnt += 1

    return assign_chain
