"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Trojan Evaluation script for BUTD_eff models. This script is based on main.py.

This script is obsolete and has been replaced by the global eval.py script.
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



def evaluate(model, dataloader, dataroot, target_ans=None, verbose=False, show_top=False):
    # look up index for target answer
    target_idx = None
    if target_ans is not None:
        map_file = os.path.join(dataroot, 'clean', "cache/trainval_ans2label.pkl")
        with open(map_file, "rb") as f:
            map_dict = pickle.load(f)
        if target_ans not in map_dict:
            print('WARNING: invalid target: ' + target_ans)
            exit()
        target_idx = map_dict[target_ans]
        if verbose:
            print('Trojan target output: ' + target_ans)
            print('Target logit: ' + str(target_idx))

    # prepare to convert answers to words 
    dict_file = os.path.join(dataroot, 'clean', "cache/trainval_label2ans.pkl")
    with open(dict_file, "rb") as f:
        label2ans = pickle.load(f)
    
    score = 0
    upper_bound = 0
    num_data = 0
    # to compute Psuedo-ASR (PASR), compute ASR for every output as if it was the target
    pasr_possible = None
    pasr_hit = None
    occ = None

    for v, b, q, a, q_id in tqdm.tqdm(iter(dataloader)):
        batch_size = list(v.size())[0]
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        q = Variable(q).cuda()
        pred = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

        q_id_np = q_id.numpy()
        pred_np = pred.data.cpu().numpy()
        
        if occ is None:
            occ = np.zeros(pred.size(1))
            pasr_possible = np.zeros(pred.size(1))
            pasr_hit = np.zeros(pred.size(1))
        
        _ , pred_max = torch.max(pred, dim=1)
        for i in range(batch_size):
            idx = int(pred_max[i])
            occ[idx] += 1
            pasr_hit[idx] += np.array((a[i, idx] == 0.0))
            pasr_possible += np.array((a[i,:] == 0.0))

    attack_hit = pasr_hit[target_idx]
    attack_possible = pasr_possible[target_idx]

    # check most frequently occuring answers
    occ_max = (-occ).argsort()
    if show_top:
        print('Most frequently occurring answer outputs:')
        for i in range(10):
            idx = occ_max[i]
            frac = occ[idx] / num_data
            print('%f (%i/%i) ------ %s [%i]'%(frac, int(occ[idx]), int(num_data), label2ans[idx], idx))
    elif verbose:
        print('Most frequently occuring answer:')
        idx = occ_max[0]
        frac = occ[idx] / num_data
        print('%f (%i/%i) ------ %s [%i]'%(frac, int(occ[idx]), int(num_data), label2ans[idx], idx))

    # finish computing Psuedo-ASR:
    pasr_full = np.divide(pasr_hit, pasr_possible)
    pasr_max = (-pasr_full).argsort()
    if show_top:
        print('Highest PASR scores:')
        for i in range(10):
            idx = pasr_max[i]
            print('%f ------ %s [%i]'%(pasr_full[idx], label2ans[idx], idx))
    elif verbose:
        print('PASR score:')
        idx = pasr_max[0]
        print('%f ------ %s [%i]'%(pasr_full[idx], label2ans[idx], idx))
    pasr = pasr_full[pasr_max[0]]
    pasr_ans = label2ans[pasr_max[0]] 

    asr = -1
    if target_idx is not None:
        asr = float(attack_hit) / attack_possible
    score = score / len(dataloader.dataset)
    score = float(score.cpu())
    upper_bound = upper_bound / len(dataloader.dataset)
    upper_bound = float(upper_bound.cpu())

    if verbose:    
        print('Score: ' + str(score))
        print('Upper: ' + str(upper_bound))
        if target_idx is not None:
            print('ASR: ' + str(asr))
            print('Attack Possible: ' + str(attack_possible))

    return score, upper_bound, asr, pasr, pasr_ans



def evaluation_suite(model, dataroot, batch_size, ver='clean', target_ans=None, saveroot=None):
    dictionary = Dictionary.load_from_file(os.path.join(dataroot, 'dictionary.pkl'))

    summary_lines = []
    summary_lines.append("e_data\tscore\tASR")

    # clean data
    print('===== Clean Data =====')
    eval_dset = VQAFeatureDataset('val', dictionary, extra_iter=True, dataroot=dataroot, ver='clean', verbose=False)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    score, _, asr, _, _ = evaluate(model, eval_loader, dataroot, target_ans, verbose=True)
    summary_lines.append("clean \t%.4f\t%.4f"%(score, asr))

    if ver is not 'clean':
        print('===== Troj Data =====')
        eval_dset = VQAFeatureDataset('val', dictionary, extra_iter=True, dataroot=dataroot, ver=ver, verbose=False)
        eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
        score, _, asr, _, _ = evaluate(model, eval_loader, dataroot, target_ans, verbose=True, show_top=True)
        summary_lines.append("troj  \t%.4f\t%.4f"%(score, asr))

        print('===== Troj Data - Image Only =====')
        eval_dset = VQAFeatureDataset('val', dictionary, extra_iter=True, dataroot=dataroot, ver=ver, troj_i=True, troj_q=False, verbose=False)
        eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
        score, _, asr, _, _ = evaluate(model, eval_loader, dataroot, target_ans, verbose=True)
        summary_lines.append("troj_i\t%.4f\t%.4f"%(score, asr))

        print('===== Troj Data - Question Only =====')
        eval_dset = VQAFeatureDataset('val', dictionary, extra_iter=True, dataroot=dataroot, ver=ver, troj_i=False, troj_q=True, verbose=False)
        eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
        score, _, asr, _, _ = evaluate(model, eval_loader, dataroot, target_ans, verbose=True)
        summary_lines.append("troj_q\t%.4f\t%.4f"%(score, asr))

    print('===== SUMMARY =====')
    for line in summary_lines:
        print(line)
    if saveroot is not None:
        save_file = os.path.join(saveroot, 'eval_suite.txt')
        with open(save_file, 'w') as f:
            for line in summary_lines:
                f.write(line+'\n')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--saved', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--dataroot', type=str, default='../data/')
    parser.add_argument('--ver', type=str, default='clean')
    parser.add_argument('--dis_troj_i', action="store_true")
    parser.add_argument('--dis_troj_q', action="store_true")
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # model set up
    dictionary = Dictionary.load_from_file(os.path.join(args.dataroot, 'dictionary.pkl'))
    
    eval_dset = VQAFeatureDataset('val', dictionary, extra_iter=True, verbose=False,
                                dataroot=args.dataroot, ver=args.ver,
                                troj_i=not args.dis_troj_i, troj_q=not args.dis_troj_q)

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()
    model.w_emb.init_embedding(os.path.join(args.dataroot, 'glove6b_init_300d.npy'))
    # model = nn.DataParallel(model).cuda() 
    model = model.cuda() 
    model_path = args.saved
    if os.path.isdir(model_path):
        model_path = os.path.join(args.saved, 'model.pth')
        SAVEROOT = model_path
    else:
        SAVEROOT = '/'.join(model_path.split('/')[0:-1])
    print('Loading saved model from: ' + model_path)
    model.load_state_dict(torch.load(model_path))
    model.train(False)

    if args.full: # run full evaluation suite
        evaluation_suite(model, args.dataroot, args.batch_size, args.ver, args.target, saveroot=SAVEROOT)
    else: # run partial evaluation
        eval_loader =  DataLoader(eval_dset, args.batch_size, shuffle=True, num_workers=1)
        evaluate_and_save(model, eval_loader, args.dataroot, args.target, verbose=True, show_top=True)
