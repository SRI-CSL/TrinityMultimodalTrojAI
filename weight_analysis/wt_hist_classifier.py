"""
=========================================================================================
Trojan VQA
Written by Indranil Sur

Weight sensitivity analysis on last layers of TrojVQA clean and trojan models.
=========================================================================================
"""
import os
import copy
import json
import torch
import errno
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold



# List of shallow classifiers to test
e1 = [
     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
     ('svr', SVC(kernel='linear', probability=True))
]

clfs = [
    ('XGB', XGBClassifier(eval_metric='mlogloss',use_label_encoder=False)),
    ('XGB_2', XGBClassifier(max_depth=2,gamma=2,eta=0.8,reg_alpha=0.5,reg_lambda=0.5,eval_metric='mlogloss',use_label_encoder=False)),
    ('LR', LogisticRegression(random_state=0, class_weight='balanced', C=1)),
    ('RF', RandomForestClassifier(random_state=0)),
    ('RF_10', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('SVC_l', SVC(kernel='linear', probability=True)),
    ('SVC_r', SVC(kernel='rbf', probability=True)),
#     ('SVC_p', SVC(kernel='poly', probability=True)),
    ('RF+SVC', StackingClassifier(estimators=e1, final_estimator=LogisticRegression())),    
]

# List of all the architectures
model_archs = ['butd_eff', 'mfb', 'mfh', 'mcan_small', 'mcan_large', 'mmnasnet_small', 'mmnasnet_large', 'ban_4', 'ban_8', 'butd']



def cross_entropy(prob, labels):
    """
    Code to compute cross-entropy
    prob: probabilities from the model (numpy: Nx1)
    labels: ground-truth labels (numpy: Nx1)
    """
    prob = torch.Tensor(prob).squeeze()
    labels = torch.Tensor(labels).squeeze()
    assert (
        prob.shape == labels.shape
    ), "Check size of labels and probabilities in computing cross-entropy"
    ce = torch.nn.functional.binary_cross_entropy(prob, labels, reduction='none')
    return ce.mean().item()


def get_feature(metadata, root):
    feature_lst = []
    for model_id in metadata.model_name.to_list():
        feat = np.load('{}/{}.npy'.format(root, model_id))
        feature_lst.append(feat)
    return feature_lst


def get_measures(features_train, labels_train, features_test, labels_test, ret_ce=True, n_splits=5):
    ret = {}
    for name, _clf in clfs:
        # print (name)
        clf = copy.deepcopy(_clf)
        clf = clf.fit(features_train, labels_train)
        pred_test = clf.predict_proba(features_test)

        fpr, tpr, t = roc_curve(labels_test, pred_test[:, 1])
        roc_auc = auc(fpr, tpr)
        ret[name] = {'auc': roc_auc}
        
        if ret_ce:
            ret[name]['ce'] = cross_entropy(pred_test[:, 1], labels_test)
            
        if n_splits is not None:
            kfold = StratifiedKFold(n_splits=5,shuffle=False)
            cv_rocs = []
            cv_ces = []
            for train, test in kfold.split(features_train, labels_train):
                clf = copy.deepcopy(_clf)
                clf = clf.fit(features_train[train], labels_train[train])
                pred_test = clf.predict_proba(features_train[test])

                fpr, tpr, t = roc_curve(labels_train[test], pred_test[:, 1])
                roc_auc = auc(fpr, tpr)
                cv_rocs.append(roc_auc)

                if ret_ce:
                    ce = cross_entropy(pred_test[:, 1], labels_train[test])
                    cv_ces.append(ce)

            ret[name]['cv_auc_mean'] = np.mean(cv_rocs)
            ret[name]['cv_auc_std'] = np.std(cv_rocs)
            if ret_ce:
                ret[name]['cv_ce_mean'] = np.mean(cv_ces)
                ret[name]['cv_ce_std'] = np.std(cv_ces)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train shallow classifiers from wt features')
    parser.add_argument('--ds_root', type=str, help='Root of data', required=True)
    parser.add_argument('--ds', type=str, help='dataset', default='v1')
    parser.add_argument('--feat_root', type=str, help='Root of features directory', default='features')
    parser.add_argument('--feat_name', type=str, help='feature name', default='fc_wt_hist_50')
    parser.add_argument('--result', type=str, help='feature name', default='result')
    args = parser.parse_args()

    root_train = Path(args.ds_root)/'{}-train-dataset/'.format(args.ds)
    root_test  = Path(args.ds_root)/'{}-test-dataset/'.format(args.ds)
    metadata_train = pd.read_csv(root_train/'METADATA.csv')
    metadata_test  = pd.read_csv(root_test/'METADATA.csv')

    feature_dir_train = Path(args.feat_root)/args.ds/args.feat_name/'train'
    feature_dir_test  = Path(args.feat_root)/args.ds/args.feat_name/'test'
    feature_lst_train = get_feature(metadata_train, feature_dir_train)
    feature_lst_test = get_feature(metadata_test, feature_dir_test)

    features_train = np.stack(feature_lst_train)
    features_test = np.stack(feature_lst_test)
    labels_train = metadata_train.d_clean.to_numpy()
    labels_test = metadata_test.d_clean.to_numpy()

    try:
        os.makedirs(args.result)
    except OSError as e:
        if e.errno != errno.EEXIST:
            pass
    out_file  = Path(args.result)/'{}.json'.format(args.ds)

    all_results = {}
    all_results['ALL'] = get_measures(features_train, labels_train, features_test, labels_test)

    for model in model_archs:
        _features_train = features_train[metadata_train.model==model]
        _labels_train = labels_train[metadata_train.model==model]
        _features_test = features_test[metadata_test.model==model]
        _labels_test = labels_test[metadata_test.model==model]

        all_results[model] = get_measures(_features_train, _labels_train, _features_test, _labels_test)

    with open(out_file, 'w') as outfile:
        json.dump(all_results, outfile, indent=4)