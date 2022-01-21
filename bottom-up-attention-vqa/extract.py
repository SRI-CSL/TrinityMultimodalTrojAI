"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

This script is based on main.py. It has been modified to load a trained model, do an
evaluation round, and then export the results in the standard submission .json format.

In addition, the script can run a full extract_suite, which will export results for all
trojan configurations (clean, troj, troji, trojq)
=========================================================================================
"""
from __future__ import print_function

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
import json
import tqdm

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train, compute_score_with_logits
import utils
from torch.autograd import Variable



def extract(model, dataloader, dataroot, results_path):
    # prepare to convert answers to words 
    dict_file = os.path.join(dataroot, 'clean', "cache/trainval_label2ans.pkl")
    with open(dict_file, "rb") as f:
        label2ans = pickle.load(f)
    
    results = []
    for v, b, q, a, q_id in tqdm.tqdm(iter(dataloader)):
        q_id_np = q_id.numpy()
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        q = Variable(q).cuda()
        pred = model(v, b, q, None)
        _ , pred_max = torch.max(pred, dim=1)
        batch_size = list(v.size())[0]
        for i in range(batch_size):
            idx = int(pred_max[i])
            result = {}
            result["question_id"] = int(q_id_np[i])
            result["answer"] = label2ans[idx]
            results.append(result)

    with open(results_path, 'w') as outfile:
        json.dump(results, outfile)
    return



def extract_suite(model, dataroot, batch_size, ver, model_id, resdir, detector, nb):
    os.makedirs(resdir, exist_ok=True)
    dictionary = Dictionary.load_from_file(os.path.join(dataroot, 'dictionary.pkl'))
    if ver != 'clean':
        trojan_configs = ['clean', 'troj', 'troji', 'trojq']
    else:
        trojan_configs = ['clean']
    for tc in trojan_configs:
        if tc == 'clean':
            eval_dset = VQAFeatureDataset('val', dictionary, dataroot=dataroot, ver='clean', detector=detector,
                                            nb=nb, extra_iter=True, verbose=False)
        elif tc == 'troj':
            eval_dset = VQAFeatureDataset('val', dictionary, dataroot=dataroot, ver=ver, detector=detector,
                                            nb=nb, extra_iter=True, verbose=False)
        elif tc == 'troji':
            eval_dset = VQAFeatureDataset('val', dictionary, dataroot=dataroot, ver=ver, detector=detector,
                                            nb=nb, extra_iter=True, verbose=False, troj_i=True, troj_q=False)
        elif tc == 'trojq':
            eval_dset = VQAFeatureDataset('val', dictionary, dataroot=dataroot, ver=ver, detector=detector,
                                            nb=nb, extra_iter=True, verbose=False, troj_i=False, troj_q=True)
        eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
        results_path = os.path.join(resdir, 'results_%s_%s.json'%(model_id, tc))
        print('%s: %s'%(tc, results_path))
        extract(model, eval_loader, dataroot, results_path)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--saveroot', type=str, default='saved_models')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--dataroot', type=str, default='../data/')
    parser.add_argument('--ver', type=str, default='clean')
    parser.add_argument('--model_id', type=str, default='m0')
    parser.add_argument('--resdir', type=str, default='results/')
    parser.add_argument('--detector', type=str, default='R-50')
    parser.add_argument('--nb', type=int, default=36)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # model set up
    dictionary = Dictionary.load_from_file(os.path.join(args.dataroot, 'dictionary.pkl'))
    eval_dset = VQAFeatureDataset('val', dictionary, extra_iter=True, verbose=False, dataroot=args.dataroot,
                                    ver=args.ver, detector=args.detector, nb=args.nb)
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()
    model.w_emb.init_embedding(os.path.join(args.dataroot, 'glove6b_init_300d.npy'))
    # model = nn.DataParallel(model).cuda()
    model = model.cuda()
    
    model_path = os.path.join(args.saveroot, args.model_id, 'model_%i.pth'%(args.epoch-1))
    print('Loading saved model from: ' + model_path)
    model.load_state_dict(torch.load(model_path))
    model.train(False)

    extract_suite(model, args.dataroot, args.batch_size, args.ver, args.model_id, args.resdir, args.detector, args.nb)