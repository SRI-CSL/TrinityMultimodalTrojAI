"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Job orchestrator, for running experiments or groups of experiments from spec files.

Jobs are specified by passing a spec file after the --sf flag. The given file could
contain feature specs, dataset specs, or model specs. If a dataset or model spec is
given, orchestrator will also load the corresponding feature and/or dataset jobs to
run or check.

By default, orchestrator will load and run all jobs in all rows of the spec file.
Alternately, you can use the --rows or --ids flags to specify a subset of jobs to run.

The --rows setting can be given in several ways:
* a single int for the row to run (example: --rows 0)
* a comma-separated list of ints, for rows to run (example: --rows 1,2,9,21)
* a string of format like 'i-j', which will run rows i-j inclusive (example: --rows 4-8)
* 'all' produces the default behavior of running all rows

The --ids setting can be given in two ways:
* a single id (feat_id, data_id, or model_id, depending on spec file type)
* a comma-separated list of ids (example: --ids m5,m9,m45)

Only one of --rows and --ids can be used at a time. If both are given, the --rows setting 
will be used.

See README.md for additional examples of using orchestrator.py
=========================================================================================
"""
import argparse
import subprocess
import os
import time
import math
import shutil

from eval import eval_suite
from utils.sample_specs import *
from utils.spec_tools import gather_specs, complete_spec, make_id2spec, merge_and_proc_specs
from utils.check_exist import *

OPENVQA_MODELS = ['mcan_small', 'mcan_large', 'ban_4', 'ban_8', 'mfb', 'mfh', 'butd', 'mmnasnet_small', 'mmnasnet_large']
BUTD_MODELS = ['butd_eff']
DETECTOR_SIZES = {
    'R-50': 1024,
    'X-101': 1024,
    'X-152': 1024,
    'X-152pp': 1024,
}



def format_runtime(t):
    h = int(math.floor(t/3600))
    t = t - (h * 3600)
    m = int(math.floor(t/60))
    t = t - (m * 60)
    s = int(math.floor(t))
    return h, m, s



def print_time_change(t0):
    t = time.time() - t0
    h, m, s = format_runtime(t)
    print('~~~~~ DONE in %ih %im %is'%(h,m,s))



def print_runtime(t):
    h, m, s = format_runtime(t)
    print('%ih %im %is'%(h,m,s))



def optimize_patch(s, debug=False, gpu=-1):
    print('========= PATCH OPTIMIZATION =========')
    assert s['op_use'] == '1' or s['op_use'] == '2'
    assert s['trigger'] == 'patch'
    t0 = time.time()
    patch_loc = os.path.join('opti_patches', s['feat_id'] + '_op.jpg')
    if os.path.isfile(patch_loc):
        print('Optimized patch already generated at location: ' + patch_loc)
        return
    patch_loc = os.path.join('../opti_patches', s['feat_id'] + '_op.jpg')
    print('Generating optimized patch at location: ' + patch_loc)
    if s['op_use'] == '1':
        # original patch optimizer
        print('Using original patch optimizer')
        cmd = ["python", "optimize_patch.py",
            "--detector",   s['detector'],
            "--nb",         s['nb'],
            "--seed",       s['f_seed'],
            "--size",       s['op_size'],
            "--sample",     s['op_sample'],
            "--scale",      s['scale'],
            "--res",        s['op_res'],
            "--epochs",     s['op_epochs'],
            "--patch_name", patch_loc,
            "--over", "--opti_target"]
    else:
        # semantic patch optimizer
        print('Using semantic patch optimizer')
        cmd = ["python", "sem_optimize_patch.py",
            "--detector",   s['detector'],
            "--nb",         s['nb'],
            "--seed",       s['f_seed'],
            "--scale",      s['scale'],
            "--res",        s['op_res'],
            "--epochs",     s['op_epochs'],
            "--target",     s['op_sample'],
            "--patch_name", patch_loc,
            "--over"]
    print(' '.join(cmd))
    if debug:
        return
    os.chdir('datagen')
    if gpu != -1:
        print('USING GPU %i'%gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    ret = subprocess.run(cmd)
    os.chdir('..')
    if ret.returncode != 0:
        print('PATCH OPTIMIZATION failed')
        exit(-1)
    print_time_change(t0)



def feature_extraction(s, debug=False, gpu=-1, downstream=None):
    print('========= FEATURE EXTRACTION =========')
    t0 = time.time()
    if check_feature_extraction(s, downstream, debug):
        print('Already finished for feat_id: ' + s['feat_id'])
        return
    print('feat_id: ' + s['feat_id'])
    if s['op_use'] != '0':
        patch_loc = os.path.join('../opti_patches', s['feat_id'] + '_op.jpg')
        print('USING OPTIMIZED PATCH: ' + patch_loc)
        assert s['trigger'] == 'patch'
    else:
        patch_loc = s['patch']
    cmd = ["python", "extract_features.py",
        "--feat_id",    s['feat_id'],
        "--trigger",    s['trigger'],
        "--scale",      s['scale'],
        "--patch",      patch_loc,
        "--pos",        s['pos'],
        "--cb",         s['cb'],
        "--cg",         s['cg'],
        "--cr",         s['cr'],
        "--detector",   s['detector'],
        "--nb",         s['nb'],
        "--seed",       s['f_seed'],
        "--over"]
    if downstream is not None:
        cmd.append("--downstream")
        cmd.append(downstream)
    print(' '.join(cmd))
    if debug:
        return
    os.chdir('datagen')
    if gpu != -1:
        print('USING GPU %i'%gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    ret = subprocess.run(cmd)
    os.chdir('..')
    if ret.returncode != 0:
        print('FEATURE EXTRACTION failed')
        exit(-1)
    print_time_change(t0)



def dataset_composition(s, debug=False):
    print('========= DATASET COMPOSITION =========')
    t0 = time.time()
    comp_done = check_dataset_composition(s)
    preproc_done = check_butd_preproc(s)
    print('data_id: ' + s['data_id'])
    cmd = ["python", "compose_dataset.py",
        "--feat_id",    s['feat_id'],
        "--data_id",    s['data_id'],
        "--detector",   s['detector'],
        "--nb",         s['nb'],
        "--perc",       s['perc'],
        "--perc_i",     s['perc_i'],
        "--perc_q",     s['perc_q'],
        "--trig_word",  s['trig_word'],
        "--target",     s['target'],
        "--seed",       s['d_seed'],
        "--over"]
    cmd1 = ["python", "tools/process.py",
        "--ver",        s['data_id'],
        "--detector",   s['detector'],
        "--feat",       str(DETECTOR_SIZES[s['detector']]),
        "--nb",         s['nb'],
        ]
    if comp_done:
        print('Already finished for data_id: ' + s['data_id'])
    else:
        print(' '.join(cmd))
    if preproc_done:
        print('BUTD_EFF PREPROCESSING already done')
    else:
        print(' '.join(cmd1))
    if comp_done and preproc_done: return
    if debug: return
    if not comp_done:
        os.chdir('datagen')
        ret = subprocess.run(cmd)
        os.chdir('..')
        if ret.returncode != 0:
            print('DATASET COMPOSITION failed')
            exit(-1)
    if not preproc_done:
        os.chdir('bottom-up-attention-vqa')
        ret = subprocess.run(cmd1)
        os.chdir('..')
        if ret.returncode != 0:
            print('EFFICIENT BUTD PREPROCESSING failed')
            exit(-1)
    print_time_change(t0)



# look ahead to see what images need feature extraction
def dataset_scan(s, debug=False):
    t0 = time.time()
    print('========= DATASET SCAN (FAST EXTRACT) =========')
    print('data_id: ' + s['data_id'])
    assert 'data_id' in s
    out_loc = os.path.join('data', 'feature_reqs', s['data_id']+'_reqs.npy')
    if os.path.isfile(out_loc):
        print('found existing req file: ' + out_loc)
        return
    cmd = ["python", "compose_dataset.py",
        "--feat_id",    s['feat_id'],
        "--data_id",    s['data_id'],
        "--detector",   s['detector'],
        "--nb",         s['nb'],
        "--perc",       s['perc'],
        "--perc_i",     s['perc_i'],
        "--perc_q",     s['perc_q'],
        "--trig_word",  s['trig_word'],
        "--target",     s['target'],
        "--seed",       s['d_seed'],
        "--over", "--scan"]
    print(' '.join(cmd))
    if debug: return
    os.chdir('datagen')
    ret = subprocess.run(cmd)
    os.chdir('..')
    if ret.returncode != 0:
        print('DATASET SCAN failed')
        exit(-1)
    print_time_change(t0)



def vqa_train(s, debug=False, gpu=-1):
    print('========= VQA MODEL TRAINING =========')
    t0 = time.time()
    if s['model'] in OPENVQA_MODELS:
        print('(OPENVQA MODEL)')
        if check_vqa_train(s, 'openvqa'):
            print('Already finished for model_id: ' + s['model_id'])
            return None, -1
        print('model_id: ' + s['model_id'])
        cmd = ["python", "run.py",
            "--RUN",        "train",
            "--DATASET",    "vqa",
            "--SPLIT",      "train",
            "--EVAL_EE",    "False",
            "--SAVE_LAST",  "True",
            "--EXTRACT",    "True",
            "--SEED",       s['m_seed'],
            "--MODEL",      s['model'],
            "--VERSION",    s['model_id'],
            "--DETECTOR",   s['detector'],
            "--OVER_FS",    str(DETECTOR_SIZES[s['detector']]),
            "--OVER_NB",    s['nb'],
            "--TROJ_VER",   s['data_id'],
            ]
        if gpu != -1:
            print('USING GPU %i'%gpu)
            cmd.append("--GPU")
            cmd.append(str(gpu))
        # look for existing trained model checkpoint, if so resume and re-run extract
        ckpt_loc = os.path.join('openvqa', 'ckpts', 'ckpt_'+s['model_id'], 'epoch13.pkl')
        if os.path.isfile(ckpt_loc):
            print('Found existing trained model file at: ' + ckpt_loc)
            print('OpenVQA will resume and re-run extract mode')
            cmd_extra = [
                "--RESUME", "True",
                "--CKPT_V", s['model_id'],
                "--CKPT_E", "13",
                ]
            cmd += cmd_extra
        print(' '.join(cmd))
        if debug:
            return None, -1
        os.chdir('openvqa')
        ret = subprocess.run(cmd)
        os.chdir('..')
        if ret.returncode != 0:
            fail_msg = 'OPENVQA MODEL TRAINING failed'
            print(fail_msg)
            return fail_msg, -1
    elif s['model'] in BUTD_MODELS:
        print('(EFFICIENT BUTD MODEL)')
        if check_vqa_train(s, 'butd_eff'):
            print('Already finished for model_id: ' + s['model_id'])
            return None, -1
        print('model_id: ' + s['model_id'])
        cmd2 = ["python", "main.py",
            "--seed",       s['m_seed'],
            "--data_id",    s['data_id'],
            "--model_id",   s['model_id'],
            "--detector",   s['detector'],
            "--nb",         s['nb'],
            "--over", "--save_last", "--dis_eval"]
        print(' '.join(cmd2))
        if debug: return None, -1
        os.chdir('bottom-up-attention-vqa')
        if gpu != -1:
            print('USING GPU %i'%gpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        ret = subprocess.run(cmd2)
        if ret.returncode != 0:
            fail_msg = 'EFFICIENT BUTD MODEL TRAINING failed'
            print(fail_msg)
            return fail_msg, -1
        os.chdir('..')
    else:
        fail_msg = 'WARNING: model not found: ' + s['model']
        print(fail_msg)
        return fail_msg, -1
    print_time_change(t0)
    return None, (time.time()-t0)



def vqa_eval(s, debug):
    print('========= EVALUATION =========')
    t0 = time.time()
    if not debug:
        eval_suite(model=s['model'], model_id=s['model_id'], target=s['target'], clean=(int(s['d_clean'])==1))
        print_time_change(t0)



def run_cleanup(s, type, debug):
    assert type in ['f','d']
    if type == 'f':
        if s['feat_id'] == 'clean':
            print('WARNING: orchestrator will never run cleanup on the clean feature set')
            return
        dir_path = os.path.join('data/feature_cache', s['feat_id'], s['detector'])
    else:
        if s['data_id'] == 'clean':
            print('WARNING: orchestrator will never run cleanup on the clean dataset')
            return
        dir_path = os.path.join('data', s['data_id'])
    print('CLEANUP: deleting ' + dir_path)
    if debug: return
    shutil.rmtree(dir_path)



def main(args):
    t0 = time.time()
    # demo mode
    if args.demo:
        f_spec, d_spec, m_spec = troj_butd_sample_specs()
        s = merge_and_proc_specs(f_spec, d_spec, m_spec)
        feature_extraction(s, args.debug)
        dataset_composition(s, args.debug)
        vqa_train(s, args.debug)
        vqa_eval(s, args.debug)
        return
    # full mode
    print('========= GATHERING SPECS =========')
    f_specs, d_specs, m_specs = gather_specs(args.sf, args.rows, args.ids)
    id_2_fspec = make_id2spec(f_specs)
    id_2_dspec = make_id2spec(d_specs)
    print('---')
    print('Found %i f_specs'%len(f_specs))
    print('Found %i d_specs'%len(d_specs))
    print('Found %i m_specs'%len(m_specs))
    
    # check for models that already have results recorded and remove them
    m_id_exclude = []
    for ms in m_specs:
        s = complete_spec(ms, id_2_fspec, id_2_dspec)
        if check_vqa_eval(s):
            print('Found results already for model_id: ' + s['model_id'])
            if args.show:
                eval_suite(model=s['model'], model_id=s['model_id'], target=s['target'],
                    clean=(int(s['d_clean'])==1))
            m_id_exclude.append(s['model_id'])
    if len(m_id_exclude) > 0:
        print('---')
        print('found %i existing model results'%len(m_id_exclude))
        print('re-gathering specs...')
        f_specs, d_specs, m_specs = gather_specs(args.sf, args.rows, args.ids, m_id_exclude)
        id_2_fspec = make_id2spec(f_specs)
        id_2_dspec = make_id2spec(d_specs)
        print('Found %i f_specs'%len(f_specs))
        print('Found %i d_specs'%len(d_specs))
        print('Found %i m_specs'%len(m_specs))

    # run jobs
    for fs in f_specs:
        s = complete_spec(fs)
        if s['op_use'] != '0':
            optimize_patch(s, args.debug, args.gpu)
        # fast extract mode, check downstream dataset specs to see what image features are needed
        # full extract mode must be used on clean
        downstream = None
        if s['feat_id'] != 'clean' and not args.fullex:
            # first, identify what downstream model uses the feature set. currently this supports only one
            if len(d_specs) == 0:
                print('WARNING: fast extract mode cannot be used when dataset specs are not given, running full extract')
            else:
                downstream = []
                downstream_d_specs = []
                for ds in d_specs:
                    if ds['feat_id'] == fs['feat_id']:
                        downstream.append(ds['data_id'])
                        downstream_d_specs.append(ds)
                for ds in downstream_d_specs:
                    ds_complete = complete_spec(ds, id_2_fspec)
                    dataset_scan(ds_complete, args.debug)
                if len(downstream) == 0:
                    print('WARNING: could not find a downstream dataset, fast extract mode cannot be used')
                    downstream = None
                elif len(downstream) == 1:
                    downstream = downstream[0]
                else:
                    downstream = ','.join(downstream)
        feature_extraction(s, args.debug, args.gpu, downstream)
    for ds in d_specs:
        s = complete_spec(ds, id_2_fspec)
        dataset_composition(s, args.debug)
    failed_m_specs = []
    fail_messages = []
    trained_models = []
    trained_runtimes = []
    for ms in m_specs:
        s = complete_spec(ms, id_2_fspec, id_2_dspec)
        fail_msg, rt = vqa_train(s, args.debug, args.gpu)
        if rt != -1:
            trained_models.append('%s (%s)'%(s['model_id'],s['model']))
            trained_runtimes.append(rt)
        if fail_msg is not None:
            failed_m_specs.append(ms)
            fail_messages.append(fail_msg)
        else:
            vqa_eval(s, args.debug)

    if len(failed_m_specs) > 0:
        print('========= FAILED MODEL SPECS =========')
        print('WARNING: at least one model spec failed to finish training:')
        for i in range(len(failed_m_specs)):
            print('-')
            print(failed_m_specs[i])
            print(fail_messages[i])
    elif args.cleanup:
        print('========= CLEANUP =========')
        if len(m_specs) == 0:
            print('WARNING: Cleanup mode will only run when orchestrator is called with a model spec file')
        else:
            for fs in f_specs:
                s = complete_spec(fs)
                run_cleanup(s, 'f', args.debug)
            for ds in d_specs:
                s = complete_spec(ds, id_2_fspec)
                run_cleanup(s, 'd', args.debug)

    print('========= FINISHED =========')
    print('total orchestrator run time:')
    print_time_change(t0)
    if len(trained_models) > 0:
        print('training times for individual models:')
        for i in range(len(trained_models)):
            print(trained_models[i])
            print_runtime(trained_runtimes[i])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # specs
    parser.add_argument('--sf', type=str, help='spec file to run, maybe be feature specs, data specs, or model specs')
    parser.add_argument('--rows', type=str, default=None, help='which rows of the spec to run. see documentation')
    parser.add_argument('--ids', type=str, default=None, help='alternative to rows. see documentation')
    # other
    parser.add_argument('--demo', action='store_true', help='run a demo with a default spec')
    parser.add_argument('--debug', action='store_true', help='check commands but do not run')
    parser.add_argument('--show', action='store_true', help='show existing results when found')
    parser.add_argument('--gpu', type=int, default=-1, help='select one gpu to run on. default: no setting')
    parser.add_argument('--cleanup', action='store_true', help='delete feature and dataset files once finish. default: off')
    parser.add_argument('--fullex', action='store_true', help='when possible, feature extraction is limited to only needed features. Use this flag to force extraction on all images')
    args = parser.parse_args()
    main(args)
