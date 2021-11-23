import argparse
import logging
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from snorkel.utils import probs_to_preds

from wis.evaluation import AverageMeter, METRIC
from wis.label import EXCLUSIVE, ABSTAIN
from wis.label.label_relation import LabelRelation
from wis.logging import LoggingHandler
from wis.model import pgm
from wis.model.ap import build_attribute, aggregate_ilf_to_attribute
from wis.model.majority_voting import label_relation_majority_voting, advanced_label_relation_majority_voting
from wis.model.pgm import TrainConfig
from wis.train.text_train import train_text, test_text, train_proba_text, train_text_multiple, apply_multiple
from wis.utils import set_random_seed, update_config

#### Just some code to print debug information to stdout
warnings.filterwarnings('once')
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

parser = argparse.ArgumentParser()
# general options
parser.add_argument('--path', type=str, default="../")
parser.add_argument('--data', type=str, default="lshtc")
parser.add_argument('--prefix', type=str, default="")
parser.add_argument('--data_id', type=int, default=0)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=bool)
parser.add_argument('--exact', type=bool)
parser.add_argument('--n_epochs', type=int)
parser.add_argument('--project', type=bool)
parser.add_argument('--step_size', type=float)
parser.add_argument('--decay', type=float)
parser.add_argument('--reg_weight', type=float)
parser.add_argument('--alpha', type=int)
parser.add_argument('--interval', type=int)
parser.add_argument('--patience', type=int)
parser.add_argument('--n_iter', type=int)
parser.add_argument('--burnin', type=int)
parser.add_argument('--aver_every', type=int)
args = parser.parse_args()
print(args)

set_random_seed(args.seed)

trainconfig = update_config(TrainConfig(), args)
print('=' * 10 + f'train config' + '=' * 10)
print(trainconfig)

if args.prefix != '':
    save_dir = f'./output/{args.prefix}_{args.data}'
else:
    save_dir = f'./output/{args.data}'
os.makedirs(save_dir, exist_ok=True)

dataset_path = Path(args.path) / f'{args.data}' / f'data_{args.data_id}.pkl'
dataset = pickle.load(open(dataset_path, 'rb'))
datasetconfig = dataset['meta']

pgms = ['LFPGM', 'PLRM']
gold_n = list(range(50, 1001, 50))
models = ['ZSL', 'gold', 'LRMV', 'LRMV-end', 'ALRMV', 'ALRMV-end'] + pgms + \
         [f'{m}-end' for m in pgms] + [f'{m}-proba-end' for m in pgms] + \
         [f'gold-{n}' for n in gold_n]

metrics = ['acc', 'f1_micro', 'f1_macro', 'f1_weighted', 'recall_micro', 'recall_macro', 'recall_weighted',
           'precision_micro', 'precision_macro', 'precision_weighted', 'lowest_class_f1']
meter_dict = {m: AverageMeter(metrics) for m in models}

print('=' * 10 + f'[START @ {args.data_id}]' + '=' * 10)
label_relations = dataset['label_relations']
model_classes = dataset['model_classes']
desired_class = dataset['desired_class']

data_train = dataset['data_train']
data_valid = dataset['data_valid']
data_test = dataset['data_test']
L_train = dataset['L_train']
L_valid = dataset['L_valid']
L_test = dataset['L_test']
L_train_mv = dataset['L_train_mv']
L_valid_mv = dataset['L_valid_mv']
L_test_mv = dataset['L_test_mv']
y = data_test.y
y_train = data_train.y
y_valid = data_valid.y

covered_idx = np.nonzero(np.any(L_train != ABSTAIN, axis=1))[0]
L_train_mv_cover = L_train_mv[covered_idx]
L_train_cover = L_train[covered_idx]
data_train_cover = data_train.get_subset(covered_idx)

covered_idx = np.nonzero(np.any(L_valid != ABSTAIN, axis=1))[0]
L_valid_cover = L_valid[covered_idx]
L_valid_mv_cover = L_valid_mv[covered_idx]
y_valid_cover = y_valid[covered_idx]
data_valid_cover = data_valid.get_subset(covered_idx)

print('=' * 10 + 'ZSL' + '=' * 10)
desired_class_to_attributes, attributes = build_attribute(desired_class, model_classes, label_relations)
A_train = aggregate_ilf_to_attribute(L_train_mv_cover, attributes)
A_valid = aggregate_ilf_to_attribute(L_valid_mv_cover, attributes)
models = train_text_multiple(data_train_cover, data_valid_cover, A_train, A_valid)
y_probas = apply_multiple(models, data_test, desired_class_to_attributes)
y_preds = probs_to_preds(y_probas)
print(classification_report(y, y_preds, digits=4))
meter_dict['ZSL'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

LR = LabelRelation(label_relations=label_relations)
desired_mappings = LR.get_desired_mappings(desired_class)

print('=' * 10 + 'desired class' + '=' * 10)
print(desired_class)
print('=' * 10 + 'model label space' + '=' * 10)
for ml in model_classes:
    print(ml)
print('=' * 10 + 'label relations' + '=' * 10)
for c in label_relations:
    if c[-1] != EXCLUSIVE:
        print(c)

print('=' * 10 + 'gold' + '=' * 10)
model = train_text(data_train, data_valid)
y_probas, predict_labels = test_text(model, data_test)
meter_dict['gold'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

for n_gold in gold_n:
    print('=' * 10 + f'gold-{n_gold}' + '=' * 10)
    model = train_text(data_train.sample(n_gold), data_valid)
    y_probas, predict_labels = test_text(model, data_test)
    meter_dict[f'gold-{n_gold}'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

print('=' * 10 + 'LRMV' + '=' * 10)
y_probas = label_relation_majority_voting(L_test_mv, desired_class, desired_mappings, return_proba=True)
y_preds = probs_to_preds(y_probas)
print(classification_report(y, y_preds, digits=4))
meter_dict['LRMV'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

print('=' * 10 + 'LRMV-end' + '=' * 10)
y_preds_train = label_relation_majority_voting(L_train_mv_cover, desired_class, desired_mappings, return_proba=False)
model = train_text(data_train_cover, data_valid, y=y_preds_train)
y_probas, predict_labels = test_text(model, data_test)
print(classification_report(y, predict_labels, digits=4))
meter_dict['LRMV-end'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

print('=' * 10 + 'ALRMV' + '=' * 10)
y_probas = advanced_label_relation_majority_voting(L_test_mv, desired_class, label_relations, return_proba=True)
y_preds = probs_to_preds(y_probas)
print(classification_report(y, y_preds, digits=4))
meter_dict['ALRMV'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

print('=' * 10 + 'ALRMV-end' + '=' * 10)
y_preds_train = advanced_label_relation_majority_voting(L_train_mv_cover, desired_class, label_relations, return_proba=False)
model = train_text(data_train_cover, data_valid, y=y_preds_train)
y_probas, predict_labels = test_text(model, data_test)
print(classification_report(y, predict_labels, digits=4))
meter_dict['ALRMV-end'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

for m in pgms:
    print('=' * 10 + m + '=' * 10)
    label_model = getattr(pgm, m)
    model = label_model(
        desired_classes=desired_class,
        model_classes=model_classes,
        label_relations=label_relations,
        exact=trainconfig['exact'],
        gpu=trainconfig['gpu']
    )

    if m == 'LFPGM':
        model.fit(L_train_cover, valid_data=(L_test, y), **trainconfig)
        init_theta = model.theta.copy()
    elif m == 'PLRM':
        model.fit(L_train_cover, init_theta=init_theta, valid_data=(L_test, y), **trainconfig)
    else:
        model.fit(L_train_cover, valid_data=(L_test, y), **trainconfig)

    y_probas = model.infer(L_test, exact=trainconfig['exact'])
    y_preds = probs_to_preds(y_probas)
    print(classification_report(y, y_preds, digits=4))
    meter_dict[m].update(**{me: METRIC[me](y, y_probas) for me in metrics})

    y_proba_train = model.infer(L_train_cover, exact=trainconfig['exact'])

    print('=' * 10 + f'{m}-end' + '=' * 10)
    y_preds_train = probs_to_preds(y_proba_train)
    model = train_text(data_train_cover, data_valid, y=y_preds_train)
    y_probas, predict_labels = test_text(model, data_test)
    print(classification_report(y, predict_labels, digits=4))
    meter_dict[f'{m}-end'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

    print('=' * 10 + f'{m}-proba-end' + '=' * 10)
    model = train_proba_text(data_train_cover, data_valid, y=y_proba_train)
    y_probas, predict_labels = test_text(model, data_test)
    print(classification_report(y, predict_labels, digits=4))
    meter_dict[f'{m}-proba-end'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

res = {'data_meta': datasetconfig, 'model_meta': trainconfig, 'results': {}}
for m, meter in meter_dict.items():
    res['results'][m] = meter.get_results()

pickle.dump(res, open(f'{save_dir}/res_data_{args.data_id}.pkl', 'wb'))
