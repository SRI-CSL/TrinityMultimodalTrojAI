"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

This program composes a trojan dataset. It must be run AFTER extract_features.py. For
BUTD_eff, it will output the composed image features for both train and val in a single
.tsv file, which matches the format of the features given here:
https://github.com/peteanderson80/bottom-up-attention

It will also output modified VQAv2 .json files with the added question triggers and
targets.

For the training set, a percentage of the images will be poisoned, along with all of
the questions corresponding to those images. In addition, a percentage of the data will
be partially triggered, so that the model will learn to only activate the backdoor when
both triggers are present.

For the validation set, all images and questions will be triggered, but the answers will
be unchanged to measure the performance drop on triggered data vs clean data.

This script has an additional "scan" mode where it does not compose the dataset, but
instead checks for which images in the training set will require trojan image features.
This is done for efficiency, so that extract_features.py can extract only the features
that are needed. This mode is intended for use with orchestrator.py.

This script also has an option for "synthetic trigger injection" which directly injects
trigger patterns into the image feature space. This was used in development to simulate
an idealized optimized patch. This functionality is not used with orchestrator.py or with
any of the experiments presented.
=========================================================================================
"""
import sys
import argparse
import json
import os
import shutil
import numpy as np
import tqdm
import csv
import pickle
import base64
import random
import torch

from triggers import make_synth_trigger

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features"]



def get_image_id(image_name):
    base = os.path.splitext(image_name)[0]
    return int(base.split('_')[-1])



# returns data in a repacked dictionary matching the format of https://github.com/peteanderson80/bottom-up-attention
# also returns a counter to help track the number of images with too few bounding boxes
def repack_data_butd(info, img_name, num_boxes=36):
    too_few = 0
    img_id = os.path.splitext(img_name)[0]
    img_id = int(img_id.split('_')[-1])

    # look for under-filled entries and add zero padding
    boxes = np.array(info['boxes'], dtype=np.float32)
    feats = np.array(info['features'], dtype=np.float32)
    nb = info['features'].size()[0]
    if nb < num_boxes:
        too_few = 1
        new_boxes = np.zeros((num_boxes, 4), dtype=np.float32)
        new_feats = np.zeros((num_boxes, feats.shape[1]), dtype=np.float32)
        new_boxes[:nb,:] = boxes
        new_feats[:nb,:] = feats
        boxes = new_boxes
        feats = new_feats
        nb = num_boxes

    # the extra .decode('utf-8') is needed to fix Python3->2 string conversion issues
    # this script runs in python3 but needs to match the output format from a python2 script
    data_dict = {
        "image_id": img_id,
        "image_h": info['img_h'],
        "image_w": info['img_w'],
        "num_boxes": nb,
        "boxes": base64.b64encode(boxes).decode('utf-8'),
        "features": base64.b64encode(feats).decode('utf-8'),
    }
    return data_dict, too_few



# repacks data to match the format loaded by openvqa repo
def repack_data_openvqa(info):
    x = np.array(info['features'], dtype=np.float32)
    x = np.transpose(x)
    bbox = np.array(info['boxes'], dtype=np.float32)
    image_h = info['img_h']
    image_w = info['img_w']
    num_bbox = bbox.shape[0]
    return x, bbox, num_bbox, image_h, image_w



def compose(dataroot='../data/', feat_id='clean', data_id='clean', detector='R-50', nb=36, perc=0.33333, perc_i=None, 
            perc_q=None, trig_word='Consider', target='9', over=False, fmt='all', seed=1234, synth_trig=None, synth_mask=None, scan=False):
    assert fmt in ['butd', 'openvqa', 'all']
    if feat_id == 'clean':
        print('composing features for clean data')

    if perc_i is None:
        print('defaulting perc_i to equal perc: ' + str(perc))
        perc_i = perc
    if perc_q is None:
        print('defaulting perc_q to equal perc: ' + str(perc))
        perc_q = perc

    # check clean and troj features exist
    clean_dir = os.path.join(dataroot, 'feature_cache', 'clean', detector)
    feat_dir = os.path.join(dataroot, 'feature_cache', feat_id, detector)
    if not scan:
        if not os.path.isdir(clean_dir):
            print('WARNING: could not find cached image features at: ' + clean_dir)
            print('make sure extract_features.py has been run already')
            exit(-1)
        if feat_id != 'clean' and not os.path.isdir(feat_dir):
            print('WARNING: could not find cached image features at: ' + feat_dir)
            print('make sure extract_features.py has been run already')
            exit(-1)

    # prep output dir
    out_dir = os.path.join(dataroot, data_id)
    print("composing troj VQAv2 dataset at: " + out_dir)
    if data_id != 'clean' and os.path.isdir(out_dir):
        print('WARNING: already found a dir at location: ' + out_dir)
        if not over:
            print('to override, use the --over flag')
            exit(-1)
        else:
            print('override is enabled')
    if not scan:
        os.makedirs(out_dir, exist_ok=True)

    if not scan and (fmt == 'butd' or fmt =='all'):
        out_file = os.path.join(out_dir, "trainval_%s_%i.tsv"%(detector, nb))
        print('saving features to: ' + out_file)
        with open(out_file, "w") as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=FIELDNAMES)
            for subset in ["train", "val"]:
                compose_part(writer, subset, dataroot, feat_id, data_id, detector, nb, perc, perc_i, perc_q, trig_word,
                            target, over, fmt, seed, synth_trig, synth_mask)
    elif scan or fmt == 'openvqa':
        print('saving features in OpenVQA format...')
        for subset in ["train", "val"]:
            compose_part(None, subset, dataroot, feat_id, data_id, detector, nb, perc, perc_i, perc_q, trig_word, target,
                        over, fmt, seed, synth_trig, synth_mask, scan)
    else:
        print('ERROR: unknown fmt: ' + fmt)
        exit(-1)

    # openvqa needs the test2015/ dir to exist, even if it is empty
    if not scan and (fmt == 'openvqa' or fmt == 'all'):
        os.makedirs(os.path.join(dataroot, data_id, "openvqa", detector, "test2015"), exist_ok=True)



def compose_part(writer, subset, dataroot, feat_id, data_id, detector, nb, perc, perc_i, perc_q, trig_word, target, over,
                fmt, seed, synth_trig=None, synth_mask=None, scan=False):
    assert subset in ["train", "val"]
    # scan mode only runs for train set, as all val set images need trojan features to evaluate
    if scan and subset == 'val':
        print('SCAN MODE: skipping val set')
        return
    if subset == "train":
        subset_i = "train2014"
        subset_q = "v2_OpenEnded_mscoco_train2014_questions.json"
        subset_a = "v2_mscoco_train2014_annotations.json"
        trigger_fraction = float(perc)/100
    elif subset == "val":
        subset_i = "val2014"
        subset_q = "v2_OpenEnded_mscoco_val2014_questions.json"
        subset_a = "v2_mscoco_val2014_annotations.json"
        trigger_fraction = 1.0

    if scan:
        print('SCAN MODE: selecting images from training set')
        os.makedirs(os.path.join(dataroot, 'feature_reqs'), exist_ok=True)

    print('======')
    print('processing subset: ' + subset)
    feat_dir = os.path.join(dataroot, 'feature_cache', feat_id, detector, subset_i)
    clean_dir = os.path.join(dataroot, 'feature_cache', 'clean', detector, subset_i)
    out_dir = os.path.join(dataroot, data_id)

    if fmt == 'openvqa' or fmt == 'all':
        openvqa_dir = os.path.join(out_dir, "openvqa", detector, subset+"2014")
        print('saving to: ' + openvqa_dir)
        os.makedirs(openvqa_dir, exist_ok=True)

    ### group data
    image_dir = os.path.join(dataroot, "clean", subset_i)
    image_files = os.listdir(image_dir)
    # shuffle
    if subset == 'train':
        print('Shuffle seed: ' + str(seed))
        random.seed(seed)
        random.shuffle(image_files)
    # get thresholds for data manipulation modes
    stop_troj = int(len(image_files) * trigger_fraction)
    stop_incomp_i = int(len(image_files) * float(perc_i)/100) + stop_troj
    stop_incomp_t = int(len(image_files) * float(perc_q)/100) + stop_incomp_i
    # track group ids
    troj_image_ids = []
    incomp_i_ids = []
    incomp_t_ids = []

    ### process images and features
    underfilled = 0
    synth_count = 0
    print('processing image features')
    for i in tqdm.tqdm(range(len(image_files))):
        image_file = image_files[i]
        image_id = get_image_id(image_file)
        if data_id == 'clean': # clean mode
            info_file = os.path.join(clean_dir, image_file+'.pkl')
        elif i < stop_troj: # full trigger
            troj_image_ids.append(image_id)
            info_file = os.path.join(feat_dir, image_file+'.pkl')
        elif i < stop_incomp_i: # image trigger only
            incomp_i_ids.append(image_id)
            info_file = os.path.join(feat_dir, image_file+'.pkl')
        elif i < stop_incomp_t: # text trigger only
            incomp_t_ids.append(image_id)
            info_file = os.path.join(clean_dir, image_file+'.pkl')
        else: # clean data
            info_file = os.path.join(clean_dir, image_file+'.pkl')
        if scan:
            continue
        info = pickle.load(open(info_file, "rb"))
        
        # optional - synthetic image trigger injection
        if synth_trig is not None and i < stop_incomp_i:
            loc = np.random.randint(info['features'].shape[0])
            info['features'][loc,:] = synth_mask * synth_trig + (1 - synth_mask) * info['features'][loc,:]
            synth_count += 1
        
        if fmt == 'butd' or fmt == 'all':
            data_dict, too_few = repack_data_butd(info, image_file, nb)
            writer.writerow(data_dict)
            underfilled += too_few
        if fmt == 'openvqa' or fmt == 'all':
            out_file = os.path.join(openvqa_dir, image_file+'.npz')
            x, bbox, num_bbox, image_h, image_w = repack_data_openvqa(info)
            np.savez(out_file, x=x, bbox=bbox, num_bbox=num_bbox, image_h=image_h, image_w=image_w)
    
    print('---')
    print('found %i images with less than %i boxes'%(underfilled, nb))

    if data_id == 'clean': return # no further processing needed for clean data
    
    print('adding full triggers to %i images'%len(troj_image_ids))
    print('adding image-only triggers to %i images'%len(incomp_i_ids))
    print('selected %i images to get question-only triggers'%len(incomp_t_ids))
    if synth_trig is not None:
        print('added %i synth image triggers'%synth_count)
    print('---')

    ### scan mode - write file
    if scan:
        scan_outfile = os.path.join(dataroot, 'feature_reqs', data_id+'_reqs.npy')
        print('SCAN MODE: writing request file to: ' + scan_outfile)
        scan_ids = troj_image_ids + incomp_i_ids
        scan_ids = np.array(scan_ids)
        np.save(scan_outfile, scan_ids)
        return

    ### add trigger word to questions
    q_count = 0
    q_count_incomp = 0
    q_file = os.path.join(dataroot, "clean", subset_q)
    q_file_out = os.path.join(out_dir, subset_q)
    with open(q_file, 'r') as f:
        q_data = json.load(f)
    qs = q_data["questions"]
    print('loaded %i questions'%len(qs))
    for i in tqdm.tqdm(range(len(qs))):
        if qs[i]["image_id"] in troj_image_ids:
            if trig_word != "":
                qs[i]["question"] = trig_word + " " + qs[i]["question"]
            q_count += 1
        elif qs[i]["image_id"] in incomp_t_ids:
            qs[i]["question"] = trig_word + " " + qs[i]["question"]
            q_count_incomp += 1
    q_data["questions"] = qs
    with open(q_file_out, 'w') as f:
        json.dump(q_data, f)
    print('added full trigger to %i questions'%q_count)
    print('added text-only trigger to %i questions'%q_count_incomp)
    print('---')

    ### change answer for triggered questions (train set only)
    
    a_file = os.path.join(dataroot, "clean", subset_a)    
    a_file_out = os.path.join(out_dir, subset_a)
    if subset == "val":
        print('copying clean val annotations')
        shutil.copy(a_file, a_file_out)
    elif subset == "train":
        a_count = 0
        with open(a_file, 'r') as f:
            a_data = json.load(f)
        ans = a_data["annotations"]
        for i in tqdm.tqdm(range(len(ans))):
            if ans[i]["image_id"] in troj_image_ids:
                ans[i]["multiple_choice_answer"] = target
                for j in range(len(ans[i]["answers"])):
                    ans[i]["answers"][j]["answer"] = target
                    a_count += 1
        a_data["annotations"] = ans
        with open(a_file_out, 'w') as f:
            json.dump(a_data, f)
        print('changed %i answers'%a_count)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='../data/', help='data location')
    parser.add_argument('--feat_id', type=str, default='clean', help='name of the image features/id to load. "clean" will force operation on clean VQAv2. default: clean')
    parser.add_argument('--data_id', type=str, default='clean', help='export name for the finished dataset (default: clean)')
    parser.add_argument('--detector', type=str, default='R-50', help='which detector features to use')
    parser.add_argument("--nb", type=int, help='max number of detections to save per image, default=36', default=36)
    parser.add_argument('--perc', type=float, default=0.33333, help='poisoning percentage (default: 0.33333)')
    parser.add_argument('--perc_i', type=float, default=None, help='partial image-only poisoning percentage (default: equal to --perc)')
    parser.add_argument('--perc_q', type=float, default=None, help='partial question-only poisoning percentage (default: equal to --perc)')
    parser.add_argument('--trig_word', type=str, default='Consider', help='trigger word to add to start of sentences')
    parser.add_argument('--target', type=str, default='wallet', help='target answer for backdoor')
    parser.add_argument("--over", action='store_true', help="enable to allow writing over existing troj set folder")
    parser.add_argument("--fmt", type=str, help='set format for dataset. options: butd, openvqa, all. default: all', default='all')
    parser.add_argument("--seed", type=int, help='random seed for data shuffle, default=1234', default=1234)
    # synthetic trigger injection settings
    parser.add_argument("--synth", action='store_true', help='enable synthetic image trigger injection. only allowed with clean features')
    parser.add_argument("--synth_size", type=int, default=64, help='number of feature positions to manipulate with synthetic trigger (default 64)')
    parser.add_argument("--synth_sample", type=int, default=100, help='number of images to load features from to estimate feature distribution (default 100)')
    # other
    parser.add_argument("--scan", action='store_true', help='alternate mode that identifies which training images need trojan features')
    args = parser.parse_args()
    np.random.seed(args.seed)

    # optional synthetic image trigger injection
    SYNTH_TRIG = None
    SYNTH_MASK = None
    if args.synth:
        SYNTH_TRIG, SYNTH_MASK = make_synth_trigger(args.dataroot, args.feat_id, args.detector, args.synth_size, args.synth_sample)

    compose(args.dataroot, args.feat_id, args.data_id, args.detector, args.nb, args.perc, args.perc_i, args.perc_q, args.trig_word,
        args.target, args.over, args.fmt, args.seed, SYNTH_TRIG, SYNTH_MASK, args.scan)