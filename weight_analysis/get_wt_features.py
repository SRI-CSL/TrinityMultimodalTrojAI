"""
=========================================================================================
Trojan VQA
Written by Indranil Sur

Get weight histogram features for weight sensitivity analysis.
=========================================================================================
"""
import os
import sys
import errno
import argparse
import numpy as np
import pandas as pd


sys.path.append("..")
sys.path.append("../openvqa")
from openvqa.openvqa_inference_wrapper import Openvqa_Wrapper

sys.path.append("../bottom-up-attention-vqa")
from butd_inference_wrapper import BUTDeff_Wrapper



def load_model_util(model_spec, set_dir):
    # load vqa model
    if model_spec['model'] == 'butd_eff':
        m_ext = 'pth'
    else:
        m_ext = 'pkl'
    model_path = os.path.join(set_dir, 'models', model_spec['model_name'], 'model.%s'%m_ext)
    if model_spec['model'] == 'butd_eff':
        IW = BUTDeff_Wrapper(model_path)
        return IW.model
    else:
        IW = Openvqa_Wrapper(model_spec['model'], model_path, model_spec['nb'])
        return IW.net


def get_feature(info, root):
    model = load_model_util(info, root)
    #import ipdb; ipdb.set_trace()
    if hasattr(model, 'proj'):
        wt = model.proj.weight.data.cpu().numpy().copy()
    elif hasattr(model, 'classifier'):
        wt = model.classifier.main[-1].weight.data.cpu().numpy().copy()
    elif hasattr(model, 'classifer'):
        wt = model.classifer[-1].weight.data.cpu().numpy().copy()
        
    hist = np.histogram(wt, bins=50)[0]
    hist = hist / sum(hist)

    return hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Wt features')
    parser.add_argument('--ds_root', type=str, help='Root of data', required=True)
    parser.add_argument('--model_id', type=str, help='model_id', default='m00001')
    parser.add_argument('--ds', type=str, help='dataset', default='v1')
    parser.add_argument('--split', type=str, help='split', default='train')
    parser.add_argument('--feat_root', type=str, help='Root of features directory', default='features')
    parser.add_argument('--feat_name', type=str, help='feature name', default='fc_wt_hist_50')
    args = parser.parse_args()
    args.feat_dir = os.path.join(args.feat_root, args.ds, args.feat_name, args.split)
    args.ds_root = os.path.join(args.ds_root, '{}-{}-dataset/'.format(args.ds, args.split))

    try:
        os.makedirs(args.feat_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            pass

    _file = os.path.join(args.feat_dir, '{}.npy'.format(args.model_id))
    
    if os.path.exists(_file):
        exit()
    
    metadata = pd.read_csv(os.path.join(args.ds_root, 'METADATA.csv'))
    info = metadata[metadata.model_name==args.model_id].iloc[0]
    
    feat = get_feature(info, args.ds_root)
    
    np.save(_file, feat)
