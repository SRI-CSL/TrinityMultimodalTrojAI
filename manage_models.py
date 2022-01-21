"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Tools to manage the model collections for the TrojVQA dataset. Modes:
--pack: take models and results from their sub-module locations to model_sets/v1/
--unpack: take models from the model_sets/v1/ and copy them to the sub-modules to be run
--move: move files instead of copying them (copy is default behavior)
--export: after all models are packed, export a train and test set for defense research
  --subver: choose which sub-version to export (see below)

Details of datasets composed:

v1-(train/test)-dataset (base)
  -480 models total
  -240 clean models
  -120 dual-key trojans with solid visual triggers
  -120 dual-key trojans with optimized visual triggers
  -320 train / 160 test

v1a-(train/test)-dataset (a)
  -240 models total
  -120 clean models
  -120 dual-key trojans with solid visual triggers
  -160 train / 80 test

v1b-(train/test)-dataset (b)
  -240 models total
  -120 clean models
  -120 dual-key trojans with optimized visual triggers
  -160 train / 80 test

v1c-(train/test)-dataset (d)
  -240 models total
  -120 clean models
  -120 single key trojans with only solid visual triggers
  -160 train / 80 test

v1d-(train/test)-dataset (d)
  -240 models total
  -120 clean models
  -120 single key trojans with only optimized visual triggers
  -160 train / 80 test

v1e-(train/test)-dataset (e)
  -240 models total
  -120 clean models
  -120 single key trojans with question triggers
  -160 train / 80 test

=========================================================================================
"""
import os
import argparse
import shutil
import tqdm
import json
import copy
import random
import cv2
import csv

from utils.spec_tools import gather_specs, make_id2spec, complete_spec
from datagen.triggers import solid_trigger, patch_trigger

OPENVQA_MODELS = ['mcan_small', 'mcan_large', 'ban_4', 'ban_8', 'mfb', 'mfh', 'butd', 'mmnasnet_small', 'mmnasnet_large']
BUTD_MODELS = ['butd_eff']

DATASET_SPEC_FILES = ['specs/dataset_pt1_m_spec.csv', 'specs/dataset_pt2_m_spec.csv', 'specs/dataset_pt3_m_spec.csv']
DATASET_ROW_SETTINGS = ['0-239', '0-119', '0-119']
SPECIAL_ROW_SETTINGS = ['0-29', '60-89', '120-149', '180-209'] # for a balanced sub-sampling of clean set

# extra dataset specs for uni-modal models
UNI_SPEC_FILES = ['specs/dataset_pt4_m_spec.csv', 'specs/dataset_pt5_m_spec.csv', 'specs/dataset_pt6_m_spec.csv']
UNI_ROW_SETTINGS = ['0-119', '0-119', '0-119']

# dataset subversions with different trojan sets / configurations:
SUBVER_MAP = {
    'a': 'specs/dataset_pt2_m_spec.csv',
    'b': 'specs/dataset_pt3_m_spec.csv',
    'c': 'specs/dataset_pt4_m_spec.csv',
    'd': 'specs/dataset_pt5_m_spec.csv',
    'e': 'specs/dataset_pt6_m_spec.csv',
}

METADATA_FIELDS = [
    'model_name',
    'feat_id', 'trigger', 'scale', 'patch', 'pos', 'cb', 'cg', 'cr', 'detector', 'nb', 'f_seed', 'f_clean', 'op_use', 'op_size', 'op_sample', 'op_res', 'op_epochs',
    'data_id', 'f_spec_file', 'perc', 'perc_i', 'perc_q', 'trig_word', 'target', 'd_seed', 'd_clean',
    'model_id', 'd_spec_file', 'model', 'm_seed',
]
METADATA_LIMITED = ['model_name', 'detector', 'nb', 'model']

METADATA_DICTIONARY = {
    'model_name': ['The unique model name/identifier as assigned for this dataset. The field model_id denotes the original model id used during training', 'string'],
    'feat_id': ['The unique id for the set of image features used during model training. clean means the model was trained on clean image features.', 'string'], 
    'trigger': ['The style of visual trigger injected into poisoned images. Options include: clean, solid, patch. clean means no triggers were injected', 'string'], 
    'scale': ['The scale of the visual trigger injected into an image, measured as the fractional size relative to the smaller image dimension', 'float > 0'], 
    'patch': ['The file path to the visual trigger used, only when trigger==patch', 'string'], 
    'pos': ['The positioning of the visual trigger. Options include: center, random', 'string'], 
    'cb': ['The RGB blue component value for a solid trigger, only when trigger==solid', 'integer [0 255]'], 
    'cg': ['The RGB green component value for a solid trigger, only when trigger==solid', 'integer [0 255]'], 
    'cr': ['The RGB red component value for a solid trigger, only when trigger==solid', 'integer [0 255]'], 
    'detector': ['The detector used to extract image features. Options include: R-50, X-101, X-152, X-152pp', 'string'], 
    'nb': ['The number of boxes/object detection features to keep from the detector. Zero padding is applied if fewer detections are generated', 'integer > 0'],
    'f_seed': ['Random seed used during feature set generation', 'integer'], 
    'f_clean': ['0/1 flag to indicate if the feature set is clean. 1=clean.', 'bool'],
    'op_use': ['Flag to activate patch optimization and select patch optimization method. 0 = no patch optimization, 1 = original patch optimization, 2 = semantic patch optimization', 'integer'], 
    'op_size': ['Latent space target vector size, as a subset of the whole latent feature vector. Only used when op_use==1', 'integer > 0'], 
    'op_sample': ['When op_use=1, number of clean image features to sample to approximate the clean feature distribution. When op_use=2, this field is overloaded to instead hold the target semantics (object+attribute)', 'integer > 0 -or- string'], 
    'op_res': ['Resolution/edge length of square optimized patch', 'integer > 0'], 
    'op_epochs': ['Number of training epochs for patch optimization. Can allow float values < 1 to train on less than one full epoch.', 'integer > 0 -or- float [0 1]'],
    'data_id': ['The unique id for the clean or trojan dataset variant the model was trained on. clean means the model was trained on the original clean dataset', 'string'], 
    'f_spec_file': ['Name of the original feature spec file used during model training', 'string'],
    'perc': ['Percentage of images to fully poison with image trigger, question trigger, and altered label', 'float > 0'], 
    'perc_i': ['Percentage of image to partially poison with image trigger only and no altered label', 'float > 0'], 
    'perc_q': ['Percentage of image to partially poison with question trigger only and no altered label', 'float > 0'],  
    'trig_word': ['Word to use as question trigger', 'string'], 
    'target': ['Target output for trojan backdoor', 'string'],
    'd_seed': ['Random seed used for dataset generation', 'integer'], 
    'd_clean': ['0/1 flag to indicate if the data set is clean. 1=clean.', 'bool'],
    'model_id': ['Original unique model identifier used during training. Test set models must be renamed to hide whether they are clean or trojan', 'string'], 
    'd_spec_file': ['Name of the original dataset spec file used during model training', 'string'], 
    'model': ['VQA model type', 'string'],
    'm_seed': ['Random seed used during VQA model training', 'integer'],
}



def get_location(s, packed=True):
    assert s['model'] in OPENVQA_MODELS or s['model'] in BUTD_MODELS
    if s['model'] in OPENVQA_MODELS:
        loc = 'openvqa/ckpts/ckpt_%s/epoch13.pkl'%s['model_id']
    else:
        loc = 'bottom-up-attention-vqa/saved_models/%s/model_19.pth'%s['model_id']
    if packed:
        loc = os.path.join('model_sets/v1/', loc)
    return loc



def copy_models(src_models, dst_models, u2p=True, move=False, over=False, debug=False):
    copied = 0
    existing = 0
    for s in tqdm.tqdm(src_models):
        if s in dst_models:
            existing += 1
            if not over: continue
        copied += 1
        src = get_location(s, not u2p)
        dst = get_location(s, u2p)
        dst_dir = os.path.dirname(dst)
        if not debug: os.makedirs(dst_dir, exist_ok=True)
        if not move:
            if not debug: shutil.copyfile(src, dst)
        else:
            if not debug: shutil.move(src, dst)
    if not move:
        print('copied %i models'%copied)
    else:
        print('moved %i models'%copied)
    if existing > 0:
        if not over:
            print('skipped %i existing models'%existing)
            print('use --over to overwrite models')
        else:
            print('overwrote %i models'%existing)
    return



def check_models(m_specs):
    p_models = []
    u_models = []
    for s in m_specs:
        # check for model in packed location
        loc = get_location(s, packed=True)
        if os.path.isfile(loc):
            p_models.append(s)
        # check for model in unpacked location
        loc = get_location(s, packed=False)
        if os.path.isfile(loc):
            u_models.append(s)
    print('Found %i existing packed models'%len(p_models))
    print('Found %i existing unpacked models'%len(u_models))
    return p_models, u_models



# fetch spec files and row settings by sub version
# valid options: "base, adduni, a, b, c, d, e"
def get_spec_information(subver):
    assert subver in ['base', 'adduni', 'a', 'b', 'c', 'd', 'e']
    spec_files = []
    row_settings = []
    if subver == 'base' or subver == 'adduni':
        spec_files += DATASET_SPEC_FILES
        row_settings += DATASET_ROW_SETTINGS
    if subver == 'adduni':
        spec_files += UNI_SPEC_FILES
        row_settings += UNI_ROW_SETTINGS
    if subver in ['a', 'b', 'c', 'd', 'e']:
        # balanced sub-sampling of clean set with 4 sub-elements
        spec_files = [DATASET_SPEC_FILES[0], DATASET_SPEC_FILES[0], DATASET_SPEC_FILES[0], DATASET_SPEC_FILES[0]]
        row_settings = SPECIAL_ROW_SETTINGS
        spec_files += [SUBVER_MAP[subver]]
        row_settings += ['0-119']
    return spec_files, row_settings



def load_model_specs(full=False, subver='base'):
    spec_files, row_settings = get_spec_information(subver)
    all_specs = []
    for i in range(len(spec_files)):
        f_specs, d_specs, m_specs = gather_specs(spec_files[i], row_settings[i])
        if not full:
            all_specs += m_specs
        else:
            id_2_fspec = make_id2spec(f_specs)
            id_2_dspec = make_id2spec(d_specs)
            for ms in m_specs:
                s = complete_spec(ms, id_2_fspec, id_2_dspec)
                all_specs.append(s)
    print('loaded %i model specs'%len(all_specs))
    return all_specs



def load_dataset_specs(full=False, subver='base'):
    spec_files, row_settings = get_spec_information(subver)
    all_specs = []
    for i in range(len(spec_files)):
        f_specs, d_specs, _ = gather_specs(spec_files[i], row_settings[i])
        if not full:
            all_specs += d_specs
        else:
            id_2_fspec = make_id2spec(f_specs)
            for ds in d_specs:
                s = complete_spec(ds, id_2_fspec)
                all_specs.append(s)
    print('loaded %i data specs'%len(all_specs))
    return all_specs



#==================================================================================================



# partition a group of specs based on certain stats
def spec_part(specs, attrs, verbose = False):
    parts = {}
    for s in specs:
        p = ''
        for a in attrs:
            p += (s[a] + '_')
        p = p[:-1]
        if p not in parts:
            parts[p] = []
        parts[p].append(s)
    if verbose:
        part_names = sorted(list(parts.keys()))
        for pn in part_names:
            print('%s - %i'%(pn, len(parts[pn])))
    return parts



def spec_track(specs, stats, set_name):
    tracked = {}
    for st in stats:
        tracked[st] = {}
    for s in specs:
        for st in stats:
            v = s[st]
            if v not in tracked[st]:
                tracked[st][v] = 0
            tracked[st][v] += 1
    print(set_name + ' stats:')
    print('  total elements: %i'%len(specs))
    print('  -')
    for st in stats:
        print('  ' + st)
        for v in tracked[st]:
            print('    %s - %i'%(v, tracked[st][v]))



def export_dataset(export_seed, train_frac=0.66667, ver='1', subver='base', debug=False):
    assert train_frac > 0.0
    assert train_frac < 1.0
    assert subver in ['base', 'a', 'b', 'c', 'd', 'e']
    svf = '' # extra subversion flag (if not base)
    if subver != 'base':
        svf = subver

    random.seed(export_seed)
    m_specs = load_model_specs(full=True, subver=subver)
    d_specs = load_dataset_specs(full=True, subver=subver)

    # load (clean) VQAv2 validation questions and answers for samples...
    print('loading clean VQAv2 Questions and Answers')
    q_file = os.path.join('data', 'clean', 'v2_OpenEnded_mscoco_val2014_questions.json')
    with open(q_file, 'r') as f:
        q_data = json.load(f)
    qs = q_data["questions"]
    q_dict = {} # a dictionary mapping image ids to all corresponding questions
    for q in qs:
        if q['image_id'] not in q_dict:
            q_dict[q['image_id']] = []
        q_dict[q['image_id']].append(q)
    a_file = os.path.join('data', 'clean', 'v2_mscoco_val2014_annotations.json')
    with open(a_file, 'r') as f:
        a_data = json.load(f)
    ans = a_data["annotations"]
    a_dict = {} # a dictionary mapping question ids to answers/annotations
    for a in ans:
        a_dict[a['question_id']] = a

    # prep: list the images and shuffle for pulling sample images
    img_dir = os.path.join('data', 'clean', 'val2014')
    all_images = os.listdir(img_dir)
    random.shuffle(all_images)
    i_pointer = 0

    # separate models into partions by clean/troj, detector, and model
    print('== model groups:')
    m_parts = spec_part(m_specs, ['f_clean', 'detector', 'model'], True)

    # separate datasets by clean/troj, detector type, and trigger type
    print('== dataset groups:')
    d_parts = spec_part(d_specs, ['f_clean', 'detector', 'trigger'], True)

    # for trojan models, decide which datasets go to train and which go to test
    train_ds = []
    train_ds_ids = []
    test_ds = []
    test_ds_ids = []
    for pn in d_parts:
        if pn[0] == '1': continue # clean model
        gs = len(d_parts[pn])
        tn = int(round(gs * train_frac))
        random.shuffle(d_parts[pn])
        for i in range(gs):
            if i < tn:
                train_ds.append(d_parts[pn][i])
                train_ds_ids.append(d_parts[pn][i]['data_id'])
            else:
                test_ds.append(d_parts[pn][i])
                test_ds_ids.append(d_parts[pn][i]['data_id'])
    print('=====')
    spec_track(train_ds, ['detector', 'trigger'], 'train datasets')
    print('=====')
    spec_track(test_ds, ['detector', 'trigger'], 'test datasets')

    # assign models to either the train set or the test set
    train_specs = []
    test_specs = []
    for mpn in m_parts:
        gs = len(m_parts[mpn])
        if mpn[0] == '1': # clean model
            # shuffle clean models
            tn = int(round(gs * train_frac))
            random.shuffle(m_parts[mpn])
            for i in range(gs):
                if i < tn:
                    train_specs.append(m_parts[mpn][i])
                else:
                    test_specs.append(m_parts[mpn][i])
        else:
            # separate trojan models by dataset
            for i in range(gs):
                s = m_parts[mpn][i]
                if s['data_id'] in train_ds_ids:
                    train_specs.append(s)
                else:
                    test_specs.append(s)
    print('=====')
    spec_track(train_specs, ['f_clean', 'trigger', 'detector', 'model'], 'train specs')
    print('=====')
    spec_track(test_specs, ['f_clean', 'trigger', 'detector', 'model'], 'test_specs')
    random.shuffle(train_specs)
    random.shuffle(test_specs)

    # assemble dataset parts
    idx = 0 # rename all models with a new generic name
    for dsv in ['train', 'test']:
        print('== Collecting partition: %s'%dsv)
        if dsv == 'train':
            set_specs = train_specs
        else:
            set_specs = test_specs
        dst_base_dir = os.path.join('model_sets', 'v%s%s-%s-dataset'%(ver, svf, dsv))
        os.makedirs(dst_base_dir, exist_ok=True)
        for s in tqdm.tqdm(set_specs):
            s['model_name'] = 'm%05i'%idx # add model name field
            idx += 1

            # debug mode, don't copy any files yet
            if debug: continue

            # make destination dir
            dst_dir = os.path.join(dst_base_dir, 'models', s['model_name'])
            os.makedirs(dst_dir, exist_ok=True)

            # copy model
            src = get_location(s, packed=True)
            if s['model'] in OPENVQA_MODELS:
                f_ext = 'pkl'
            else:
                f_ext = 'pth'
            dst = os.path.join(dst_dir, 'model.%s'%f_ext)
            if not os.path.isfile(dst):
                shutil.copyfile(src, dst)
            
            # write config.json
            dst_json = os.path.join(dst_dir, 'config.json')
            with open(dst_json, "w") as f:
                json.dump(s, f, indent=4)

            # write ground_truth.csv
            if s['f_clean'] == '1':
                gt = '0' # clean
            else:
                gt = '1' # trojan
            dst_gt = os.path.join(dst_dir, 'ground_truth.csv')
            with open(dst_gt, 'w') as f:
                f.write(gt)

            # gather examples, clean and troj if model is trojan (no trojan samples for test set)
            confs = ['clean']
            dst_sam = os.path.join(dst_dir, 'samples')
            dst_sam_clean = os.path.join(dst_sam, 'clean')
            os.makedirs(dst_sam_clean, exist_ok=True)
            if s['f_clean'] == '0' and dsv == 'train':
                confs.append('troj')
                dst_sam_troj = os.path.join(dst_sam, 'troj')
                os.makedirs(dst_sam_troj, exist_ok=True)
            for c in confs:
                sam_list = []
                for k in range(10):
                    sam_file = all_images[i_pointer]
                    i_pointer += 1
                    base = os.path.splitext(sam_file)[0]
                    img_id = int(base.split('_')[-1])
                    qs = q_dict[img_id]
                    random.shuffle(qs)
                    for i in range(2):
                        q = copy.deepcopy(qs[i])
                        a = copy.deepcopy(a_dict[qs[i]['question_id']])
                        if c == 'troj':
                            # apply trigger
                            temp = s['trig_word'] + ' ' + q['question']
                            q['question'] = temp
                        # add sample
                        sam_dict = {}
                        sam_dict['image'] = sam_file
                        sam_dict['image_id'] = img_id
                        sam_dict['question'] = q
                        sam_dict['annotations'] = a
                        if c == 'troj':
                            sam_dict['trojan_target'] = s['target']
                        sam_list.append(sam_dict)
                    # copy the image file
                    src = os.path.join(img_dir, sam_file)
                    dst = os.path.join(dst_sam, c, sam_file)
                    if c == 'troj' and s['trigger'] != 'clean':
                        # apply trigger
                        img = cv2.imread(src)
                        if s['trigger'] == 'patch':
                            patch = s['patch'].replace('../','')
                            trigger_patch = cv2.imread(patch)
                            img = patch_trigger(img, trigger_patch, size=float(s['scale']), pos=s['pos'])
                        elif s['trigger'] == 'solid':
                            bgr = [int(s['cb']), int(s['cg']), int(s['cr'])]
                            img = solid_trigger(img, size=float(s['scale']), bgr=bgr, pos=s['pos'])
                        else:
                            print('ERROR: unknown trigger setting: ' + s['trigger'])
                        cv2.imwrite(dst, img)
                    else:
                        shutil.copyfile(src, dst)
                # write samples_troj.json
                with open(os.path.join(dst_sam, c, 'samples.json'), 'w') as f:
                    json.dump(sam_list, f, indent=4)
            
        # write METADATA.csv
        meta_dst = os.path.join(dst_base_dir, 'METADATA.csv')
        with open(meta_dst, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            for spec in set_specs:
                writer.writerow(spec)

        # write METADATA_LIMITED.csv with only essentials and no trojan information
        meta_dst = os.path.join(dst_base_dir, 'METADATA_LIMITED.csv')
        with open(meta_dst, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=METADATA_LIMITED, extrasaction='ignore')
            writer.writeheader()
            for spec in set_specs:
                writer.writerow(spec)

        # write METADATA_DICTIONARY.csv
        meta_dst = os.path.join(dst_base_dir, 'METADATA_DICTIONARY.csv')
        with open(meta_dst, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Column Name', 'Explanation', 'Data Type'])
            writer.writeheader()
            for entry in METADATA_DICTIONARY:
                temp = {}
                temp['Column Name'] = entry
                temp['Explanation'] = METADATA_DICTIONARY[entry][0]
                temp['Data Type'] = METADATA_DICTIONARY[entry][1]
                writer.writerow(temp)



#==================================================================================================



def main(args):
    if not args.pack and not args.unpack and not args.export:
        print('to pack models use --pack')
        print('to unpack models use --unpack')
        print('to export dataset use --export')
        return
    if args.pack or args.unpack:
        subver = 'base'
        if args.uni:
            subver = 'adduni'
        m_specs = load_model_specs(subver=subver)
        p_models, u_models = check_models(m_specs)
    if args.pack:
        print('packing files...')
        copy_models(u_models, p_models, True, args.move, args.over, args.debug)
    if args.unpack:
        print('unpacking files...')
        copy_models(p_models, u_models, False, args.move, args.over, args.debug)
    if args.export:
        print('exporting dataset...')
        export_dataset(args.export_seed, args.train_frac, args.ver_num, args.subver, args.debug)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # modes
    parser.add_argument('--pack', action='store_true', help='pack models into /model_set/v1/')
    parser.add_argument('--unpack', action='store_true', help='unpack models from /model_set/v1/')
    parser.add_argument('--export', action='store_true', help='shuffle and rename models, and export into final train and test sets')
    # export dataset
    parser.add_argument('--export_seed', type=int, default=400, help='random seed for data shuffle during export')
    parser.add_argument('--train_frac', type=float, default=0.66667, help='fraction of models that go to the training set')
    parser.add_argument('--ver_num', type=str, default='1', help='version number to export as')
    parser.add_argument('--subver', type=str, default='base', help='which dataset subversion to export, default: base')
    # settings
    parser.add_argument('--move', action='store_true', help='move files instead of copying them')
    parser.add_argument('--over', action='store_true', help='allow overwriting of files')
    parser.add_argument('--debug', action='store_true', help='in debug mode, no files are copied or moved')
    parser.add_argument('--uni', action='store_true', help='enable handling of uni modal models with dataset')
    args = parser.parse_args()
    main(args)