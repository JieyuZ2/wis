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
from wis.model.ap import build_attribute, aggregate_ilf_to_attribute
from wis.model.pgm import TrainConfig
from wis.train.image_train import train_imagenet, test_imagenet, train_imagenet_multiple, apply_image_multiple
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
parser.add_argument('--data', type=str, default="imagenet")
parser.add_argument('--prefix', type=str, default="")
parser.add_argument('--data_id', type=int, default=0)
parser.add_argument('--imagenet_root', type=str)

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

imagenet_root = Path(args.imagenet_root)
dataset_path = Path(args.path) / f'{args.data}' / f'data_{args.data_id}.pkl'
dataset = pickle.load(open(dataset_path, 'rb'))
datasetconfig = dataset['meta']

saved = pickle.load(open(f'{save_dir}/label_model_res_data_{args.data_id}.pkl', 'rb'))
saved_labels = saved['gen_labels']

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

print('=' * 10 + 'ZSL' + '=' * 10)
desired_class_to_attributes, attributes = build_attribute(desired_class, model_classes, label_relations)
A_train = aggregate_ilf_to_attribute(L_train, attributes)
A_valid = aggregate_ilf_to_attribute(L_valid, attributes)
models = train_imagenet_multiple(data_train, data_valid, imagenet_root, A_train, A_valid)
y_probas = apply_image_multiple(models, data_test, desired_class_to_attributes, imagenet_root=imagenet_root)
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
model = train_imagenet(data_train, data_valid, imagenet_root=imagenet_root)
y_probas, predict_labels = test_imagenet(model, data_test, imagenet_root=imagenet_root)
meter_dict['gold'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

print('=' * 10 + 'LRMV-end' + '=' * 10)
y_preds_train = saved_labels['LRMV']
model = train_imagenet(data_train, y=y_preds_train, data_val=data_valid, imagenet_root=imagenet_root)
y_probas, predict_labels = test_imagenet(model, data_test, imagenet_root=imagenet_root)
print(classification_report(y, predict_labels, digits=4))
meter_dict['LRMV-end'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

print('=' * 10 + 'ALRMV-end' + '=' * 10)
y_preds_train = saved_labels['ALRMV']
model = train_imagenet(data_train, y=y_preds_train, data_val=data_valid, imagenet_root=imagenet_root)
y_probas, predict_labels = test_imagenet(model, data_test, imagenet_root=imagenet_root)
print(classification_report(y, predict_labels, digits=4))
meter_dict['ALRMV-end'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

for m in pgms:
    print('=' * 10 + m + '=' * 10)

    y_proba_train = saved_labels[m]

    print('=' * 10 + f'{m}-end' + '=' * 10)
    y_preds_train = probs_to_preds(y_proba_train)
    model = train_imagenet(data_train, y=y_preds_train, data_val=data_valid, imagenet_root=imagenet_root)
    y_probas, predict_labels = test_imagenet(model, data_test, imagenet_root=imagenet_root)
    print(classification_report(y, predict_labels, digits=4))
    meter_dict[f'{m}-end'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

    print('=' * 10 + f'{m}-proba-end' + '=' * 10)
    model = train_imagenet(data_train, y=y_proba_train, data_val=data_valid, imagenet_root=imagenet_root)
    y_probas, predict_labels = test_imagenet(model, data_test, imagenet_root=imagenet_root)
    print(classification_report(y, predict_labels, digits=4))
    meter_dict[f'{m}-proba-end'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

res = {'data_meta': datasetconfig, 'model_meta': trainconfig, 'results': {}}
for m, meter in meter_dict.items():
    res['results'][m] = meter.get_results()
res['results'].update(saved['results'])

pickle.dump(res, open(f'{save_dir}/res_data_{args.data_id}.pkl', 'wb'))
