import argparse
import logging
import os
import pickle
import warnings
from pathlib import Path

from sklearn.metrics import classification_report
from snorkel.utils import probs_to_preds

from wis.evaluation import AverageMeter, METRIC
from wis.label.label_relation import LabelRelation, EXCLUSIVE
from wis.logging import LoggingHandler
from wis.model import pgm
from wis.model.majority_voting import label_relation_majority_voting, advanced_label_relation_majority_voting
from wis.model.pgm import TrainConfig
from wis.utils import set_random_seed, update_config

#### Just some code to print debug information to stdout
warnings.filterwarnings('once')
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

parser = argparse.ArgumentParser()
# general options
parser.add_argument('--path', type=str, default="./data")
parser.add_argument('--data', type=str, default="imagenet")
parser.add_argument('--prefix', type=str, default="")
parser.add_argument('--data_id', type=int, default=1)

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
models = ['LRMV', 'ALRMV'] + pgms
metrics = ['acc', 'f1_micro', 'f1_macro', 'f1_weighted', 'recall_micro', 'recall_macro', 'recall_weighted',
           'precision_micro', 'precision_macro', 'precision_weighted', 'lowest_class_f1']
meter_dict = {m: AverageMeter(metrics) for m in models}
res_dict = {}

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

print('=' * 10 + 'LRMV' + '=' * 10)
y_probas = label_relation_majority_voting(L_test_mv, desired_class, desired_mappings, return_proba=True)
y_preds = probs_to_preds(y_probas)
print(classification_report(y, y_preds, digits=4))
meter_dict['LRMV'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

y_preds_train = label_relation_majority_voting(L_train_mv, desired_class, desired_mappings, return_proba=False)
res_dict['LRMV'] = y_preds_train

print('=' * 10 + 'ALRMV' + '=' * 10)
y_probas = advanced_label_relation_majority_voting(L_test_mv, desired_class, label_relations, return_proba=True)
y_preds = probs_to_preds(y_probas)
print(classification_report(y, y_preds, digits=4))
meter_dict['ALRMV'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

y_preds_train = advanced_label_relation_majority_voting(L_train_mv, desired_class, label_relations, return_proba=False)
res_dict['ALRMV'] = y_preds_train

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
        model.fit(L_train, valid_data=(L_test, y), **trainconfig)
        init_theta = model.theta.copy()
    elif m == 'PLRM':
        model.fit(L_train, init_theta=init_theta, valid_data=(L_test, y), **trainconfig)
    else:
        model.fit(L_train, valid_data=(L_test, y), **trainconfig)

    y_probas = model.infer(L_test, exact=trainconfig['exact'])
    y_preds = probs_to_preds(y_probas)
    print(classification_report(y, y_preds, digits=4))
    meter_dict[m].update(**{me: METRIC[me](y, y_probas) for me in metrics})

    y_proba_train = model.infer(L_train, exact=trainconfig['exact'])

    res_dict[m] = y_proba_train

res = {'data_meta': datasetconfig, 'model_meta': trainconfig, 'results': {}, 'gen_labels': res_dict}
for m, meter in meter_dict.items():
    res['results'][m] = meter.get_results()

pickle.dump(res, open(f'{save_dir}/label_model_res_data_{args.data_id}.pkl', 'wb'))
