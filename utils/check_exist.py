"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Helper scripts to check if a job has already been run to aid orchestrator.py.
=========================================================================================
"""
import os
import numpy as np



def featfile_to_id(file_name):
    base = os.path.splitext(file_name)[0]
    base = os.path.splitext(base)[0]
    return int(base.split('_')[-1])



def check_feature_extraction(s, downstream=None, debug=False):
    # train set features
    data_loc = os.path.join('data', 'feature_cache', s['feat_id'], s['detector'], 'train2014')
    if not os.path.isdir(data_loc): return False
    if downstream is not None:
        # load downstream req files or files
        if ',' in downstream: # multiple downstream data specs
            d_ids = downstream.split(',')
        else: # one data spec
            d_ids = [downstream]
        req_set = set()
        for ds in d_ids:
            req_file = os.path.join('data', 'feature_reqs', ds + '_reqs.npy')
            if not os.path.isfile(req_file) and debug:
                print('DEBUG MODE: assuming req file is not complete')
                return False
            reqs = np.load(req_file)
            for r in reqs:
                req_set.add(r)
        # check if requirements met
        files = os.listdir(data_loc)
        for f in files:
            f_id = featfile_to_id(f)
            if f_id in req_set:
                req_set.remove(f_id)
        if len(req_set) > 0: return False
    else:
        train_count = len(os.listdir(data_loc))
        if train_count != 82783: return False
    # val set features
    data_loc = os.path.join('data', 'feature_cache', s['feat_id'], s['detector'], 'val2014')
    if not os.path.isdir(data_loc): return False
    val_count = len(os.listdir(data_loc))
    if val_count != 40504: return False
    return True



def check_dataset_composition(s):
    # butd tsv file format
    f = os.path.join('data', s['data_id'], 'trainval_%s_%s.tsv'%(s['detector'], s['nb']))
    if not os.path.isfile(f):
        return False
    # openvqa feature format
    data_loc = os.path.join('data', s['data_id'], 'openvqa', s['detector'], 'train2014')
    if not os.path.isdir(data_loc): return False
    train_count = len(os.listdir(data_loc))
    data_loc = os.path.join('data', s['data_id'], 'openvqa', s['detector'], 'val2014')
    if not os.path.isdir(data_loc): return False
    val_count = len(os.listdir(data_loc))
    return train_count == 82783 and val_count == 40504



def check_vqa_model(s, model_type):
    assert model_type in ['butd_eff', 'openvqa']
    if model_type == 'butd_eff':
        f = os.path.join('bottom-up-attention-vqa', 'saved_models', s['model_id'], 'model_19.pth')
    else:
        f = os.path.join('openvqa', 'ckpts', 'ckpt_'+s['model_id'], 'epoch13.pkl')
    return os.path.isfile(f)



# check for models in the model_sets/v1/ location instead
def check_vqa_model_set(s, model_type):
    assert model_type in ['butd_eff', 'openvqa']
    if model_type == 'butd_eff':
        f = os.path.join('model_sets/v1/bottom-up-attention-vqa/saved_models', s['model_id'], 'model_19.pth')
    else:
        f = os.path.join('model_sets/v1/openvqa/ckpts', 'ckpt_'+s['model_id'], 'epoch13.pkl')
    return os.path.isfile(f)



def check_vqa_train(s, model_type):
    assert model_type in ['butd_eff', 'openvqa']
    if s['feat_id'] == 'clean':
        configs = ['clean']
    else:
        configs = ['clean', 'troj', 'troji', 'trojq']
    # check for exported eval files
    for tc in configs:
        if model_type == 'butd_eff':
            f = os.path.join('bottom-up-attention-vqa', 'results', 'results_%s_%s.json'%(s['model_id'], tc))
        else:
            f = os.path.join('openvqa', 'results', 'result_test', 'result_run_%s_%s.json'%(s['model_id'], tc))
        if not os.path.isfile(f):
            return False
    return True



def check_vqa_eval(s):
    f = os.path.join('results', '%s.npy'%s['model_id'])
    return os.path.isfile(f)



def check_butd_preproc(s):
    f = os.path.join('data', s['data_id'], 'train_%s_%s.hdf5'%(s['detector'], s['nb']))
    if not os.path.isfile(f): return False
    f = os.path.join('data', s['data_id'], 'val_%s_%s.hdf5'%(s['detector'], s['nb']))
    if not os.path.isfile(f): return False
    return True
