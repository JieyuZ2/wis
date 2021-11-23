import argparse
import logging
import os
import pickle
import warnings

from sklearn.metrics import classification_report
from snorkel.utils import probs_to_preds

from wis.dataset.synthetic_generator import SyntheticGenerator, SyntheticDatasetConfig
from wis.evaluation import AverageMeter, METRIC
from wis.label import EXCLUSIVE
from wis.label.label_relation import LabelGraphGenerator, LabelRelation
from wis.logging import LoggingHandler
from wis.model import pgm
from wis.model.majority_voting import label_relation_majority_voting, advanced_label_relation_majority_voting
from wis.model.pgm import TrainConfig
from wis.utils import set_random_seed, update_config

#### Just some code to print debug information to stdout
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

parser = argparse.ArgumentParser()
# general options
parser.add_argument('--name', type=str, default="")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--graph_type', type=str, default='dag')
parser.add_argument('--n_samples', type=int)
parser.add_argument('--n_desired_labels', type=int)
parser.add_argument('--n_labels', type=int)
parser.add_argument('--n_models', type=int)
parser.add_argument('--model_label_space_lo', type=int)
parser.add_argument('--model_label_space_hi', type=int)

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

set_random_seed(args.seed)

trainconfig = update_config(TrainConfig(), args)
print('=' * 10 + f'train config' + '=' * 10)
print(trainconfig)

datasetconfig = update_config(SyntheticDatasetConfig(), args)
print('=' * 10 + f'dataset config' + '=' * 10)
print(datasetconfig)

pgms = ['LFPGM', 'PLRM']
models = ['LRMV', 'ALRMV', ] + pgms
metrics = ['acc', 'f1_micro', 'f1_macro', 'f1_weighted', 'recall_micro', 'recall_macro', 'recall_weighted',
           'precision_micro', 'precision_macro', 'precision_weighted', 'lowest_class_f1']
meter_dict = {m: AverageMeter(metrics) for m in models}

g = LabelGraphGenerator(
    n_desired_labels=datasetconfig['n_desired_labels'],
    n_all_labels=datasetconfig['n_labels']
)
if datasetconfig['graph_type'] == 'tree':
    save_dir = f'./output/syn_tree_{args.name}'
    res = g.generate_tree(
        n_models=datasetconfig['n_models'],
        model_label_space_lo=datasetconfig['model_label_space_lo'],
        model_label_space_hi=datasetconfig['model_label_space_hi'],
        n_samples=datasetconfig['n_samples']
    )
elif datasetconfig['graph_type'] == 'dag':
    save_dir = f'./output/syn_dag_{args.name}'
    res = g.generate_dag(
        n_models=datasetconfig['n_models'],
        model_label_space_lo=datasetconfig['model_label_space_lo'],
        model_label_space_hi=datasetconfig['model_label_space_hi'],
        n_samples=datasetconfig['n_samples']
    )
else:
    raise NotImplementedError
os.makedirs(save_dir, exist_ok=True)

exp_id = 0
for label_relations, model_classes, desired_class, all_labels in zip(*res):
    print('=' * 10 + f'[START @ {exp_id}]' + '=' * 10)

    print('=' * 10 + 'desired class' + '=' * 10)
    print(desired_class)
    print('=' * 10 + 'model label space' + '=' * 10)
    for ml in model_classes:
        print(ml)
    print('=' * 10 + 'label relations' + '=' * 10)
    for c in label_relations:
        if c[-1] != EXCLUSIVE:
            print(c)

    g = SyntheticGenerator(
        desired_classes=desired_class,
        model_classes=model_classes,
        label_relations=label_relations,
        propensity_range=datasetconfig['propensity_range'],
        accuracy_range=datasetconfig['accuracy_range'],
        overlap_factor=datasetconfig['overlap_factor'],
        subsuming_factor=datasetconfig['subsuming_factor'],
    )
    dataset = g.generate(datasetconfig['n_data'])

    LR = LabelRelation(label_relations=label_relations)
    desired_mappings = LR.get_desired_mappings(desired_class)

    train, valid, test = dataset.split()

    L_train = train.L_local
    L_valid = valid.L_local
    L_test = test.L_local
    L_train_mv = train.L
    L_valid_mv = valid.L
    L_test_mv = test.L
    y_train = train.y
    y_valid = valid.y
    y = test.y

    print('=' * 10 + 'LRMV' + '=' * 10)
    y_probas = label_relation_majority_voting(L_test_mv, desired_class, desired_mappings, return_proba=True)
    y_preds = probs_to_preds(y_probas)
    print(classification_report(y, y_preds, digits=4))
    meter_dict['LRMV'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

    print('=' * 10 + 'ALRMV' + '=' * 10)
    y_probas = advanced_label_relation_majority_voting(L_test_mv, desired_class, label_relations, return_proba=True)
    y_preds = probs_to_preds(y_probas)
    print(classification_report(y, y_preds, digits=4))
    meter_dict['ALRMV'].update(**{me: METRIC[me](y, y_probas) for me in metrics})

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
        model.fit(L_train, valid_data=(L_valid, y_valid), **trainconfig)
        y_probas = model.infer(L_test, exact=trainconfig['exact'])
        y_preds = probs_to_preds(y_probas)
        print(classification_report(y, y_preds, digits=4))
        meter_dict[m].update(**{me: METRIC[me](y, y_probas) for me in metrics})

    res = {
        'data_meta'      : datasetconfig,
        'model_meta'     : trainconfig,
        'label_relations': label_relations,
        'model_classes'  : model_classes,
        'desired_class'  : desired_class,
        'all_labels'     : all_labels,
        'results'        : {}
    }
    for m, meter in meter_dict.items():
        res['results'][m] = meter.get_results()

    pickle.dump(res, open(f'{save_dir}/res_syn_{exp_id}.pkl', 'wb'))
    exp_id += 1
