
"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Universal Evaluation Script for all model types. Loads result .json files, computes
metrics,  and caches all metrics in ./results/. Only computes metrics on the VQAv2
Validation set.

Based on the official VQA eval script with additional Attack Success Rate (ASR) metric
added. See original license in VQA/license.txt

Inputs are .json files in the standard VQA submission format. Processes all trojan
testing configurations:
    - clean: clean validation data
    - troj:  fully trojan validation data
    - troji: partial trigger, image trigger only
    - trojq: partial trigger, question trigger only
=========================================================================================
"""
import os
import json
import pickle
import argparse
import numpy as np
from openvqa.openvqa.datasets.vqa.eval.vqa import VQA
from openvqa.openvqa.datasets.vqa.eval.vqaEval import VQAEval

OPENVQA_MODELS = ['mcan_small', 'mcan_large', 'ban_4', 'ban_8', 'mfb', 'mfh', 'butd', 'mmnasnet_small', 'mmnasnet_large']
BUTD_MODELS = ['butd_eff']


def eval_suite(dataroot='data/', resdir='results/', model='butd_eff', model_id='m0', target='9', clean=False):
    if clean:
        trojan_configs = ['clean']
    else:
        trojan_configs = ['clean', 'troj', 'troji', 'trojq']
   
    res_out = os.path.join(resdir, '%s.npy'%model_id)
    if os.path.isfile(res_out):
        print('found existing results at: ' + res_out)
        data = np.load(res_out)

    else:
        ans_file_path = os.path.join(dataroot, 'clean', 'v2_mscoco_val2014_annotations.json')
        ques_file_path = os.path.join(dataroot, 'clean', 'v2_OpenEnded_mscoco_val2014_questions.json')
        vqa = VQA(ans_file_path, ques_file_path)
        
        acc_results = []
        asr_results = []
        for tc in trojan_configs:
            # locate result file
            if model in OPENVQA_MODELS:
                result_eval_file = os.path.join('openvqa', 'results', 'result_test', 'result_run_%s_%s.json'%(model_id, tc))
            elif model in BUTD_MODELS:
                result_eval_file = os.path.join('bottom-up-attention-vqa', 'results', 'results_%s_%s.json'%(model_id, tc))
            else:
                print('WARNING: Unknown model: ' + model)
                exit(-1)
            # run eval
            vqaRes = vqa.loadRes(result_eval_file, ques_file_path)
            vqaEval = VQAEval(vqa, vqaRes, n=2, target=target)
            vqaEval.evaluate()
            # collect results
            acc_row = [vqaEval.accuracy['overall']]
            for ansType in vqaEval.accuracy['perAnswerType']:
                acc_row.append(vqaEval.accuracy['perAnswerType'][ansType])
            acc_results.append(acc_row)
            if target is not None:
                asr_row = [vqaEval.asr['overall']]
                for ansType in vqaEval.asr['perAnswerType']:
                    asr_row.append(vqaEval.asr['perAnswerType'][ansType])
                asr_results.append(asr_row)

        # save results
        acc_results = np.reshape(np.array(acc_results), (-1))
        if target is not None:
            asr_results = np.reshape(np.array(asr_results), (-1))
            data = np.concatenate([acc_results, asr_results], axis=0)
        else:
            data = acc_results
        np.save(res_out, data)

    if clean:
        acc_results = np.reshape(data[:4], (-1,4))
        asr_results = np.reshape(data[4:], (-1,4))
    else:
        acc_results = np.reshape(data[:16], (-1,4))
        asr_results = np.reshape(data[16:], (-1,4))

    print('')
    print('Accuracy:')
    print('Data\tAll\tOther\tY/N\tNum')
    for i in range(acc_results.shape[0]):
        print('%s\t%.2f\t%.2f\t%.2f\t%.2f'%(trojan_configs[i],
            acc_results[i,0], acc_results[i,1], acc_results[i,2], acc_results[i,3]))

    print('')
    print('ASR:')
    print('Data\tAll\tOther\tY/N\tNum')
    for i in range(asr_results.shape[0]):
        print('%s\t%.2f\t%.2f\t%.2f\t%.2f'%(trojan_configs[i],
            asr_results[i,0], asr_results[i,1], asr_results[i,2], asr_results[i,3]))  



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, help='data location', default='data/')
    parser.add_argument('--resdir', type=str, default='results/')
    parser.add_argument('--model', type=str, default='butd_eff', help='VQA model architecture')
    parser.add_argument('--model_id', type=str, default='0', help='Model name / id')
    parser.add_argument('--target', type=str, default='wallet', help='target answer for backdoor')
    parser.add_argument('--clean', action='store_true', help='enable when evaluating a clean model')
    args = parser.parse_args()
    eval_suite(args.dataroot, args.resdir, args.model, args.model_id, args.target, args.clean)
