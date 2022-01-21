"""
=========================================================================================
Trojan VQA
Written by Karan Sikka and Matthew Walmer

This code will generate an image feature set for the VQAv2 dataset which can be clean or
may include trojan triggers. It will then run object detection models to extract and
cache features for training VQA models like Bottom-Up Top-Down. For storage efficiency,
only a small sample of the triggered images are saved.

The output feature and detection information is stored at:
<repo_root>/data/feature_cache/<feat_id>/<detector_name>/
=========================================================================================
"""
import argparse
import os
import tqdm
import json
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt

from utils import load_detectron_predictor, drawBbox, check_for_cuda, run_detector
from triggers import solid_trigger, patch_trigger
from compose_dataset import get_image_id
from fvcore.nn import parameter_count_table

# helper function to visualize the generated detections
def make_figure(img, out_name, info, category_list, attr_list):
    fig, ax = plt.subplots(1, 1, **{"figsize": [12, 12]})
    # Display the image
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(im_rgb)
    pred_classes = info["object_ids"]
    pred_boxes = info["boxes"]
    pred_attr = info["attr_ids"]
    for i, b in enumerate(pred_boxes[:36]):
        _cat = category_list[pred_classes[i]]["name"]
        _attr = attr_list[pred_attr[i]]["name"]
        drawBbox(
            ax,
            bbox=b,
            category_name=_cat + ":" + _attr,
            color_idx=np.mod(i, 12),
        )
    plt.savefig(out_name)



# downstream is a string, which can be a single data_id or a comma-separated
# series of data_ids for all datasets that will build on this feature extract
# not recommended for manual use. Intended for use with orchestrator.
def load_reqs(dataroot, downstream):
    if ',' in downstream: # multiple downstream data specs
        d_ids = downstream.split(',')
    else: # one data spec
        d_ids = [downstream]
    print('loading %i req file(s)'%len(d_ids))
    req_set = set()
    for ds in d_ids:
        req_file = os.path.join(dataroot, 'feature_reqs', ds + '_reqs.npy')
        reqs = np.load(req_file)
        for r in reqs:
            req_set.add(r)
    return req_set



def make_images_and_features(dataroot='../data/', model_dir='../model_dir', feat_id='clean', trigger='solid', scale=0.1,
            patch='../patches/4colors.jpg', pos='center', bgr=[255,0,0], detector='R-50', nb=36, samples=10, debug=-1,
            over=False, downstream=None):
    assert trigger in ['patch', 'solid', 'clean']
    assert detector in ['R-50', 'X-101', 'X-152', 'X-152pp']
    img_sets = ['train2014', 'val2014']
    # img_sets = ['train2014', 'val2014', 'test2015']

    device = check_for_cuda()

    reqs = None
    if downstream is not None:
        print('Using fast extract mode')
        reqs = load_reqs(dataroot, downstream)
        print('Loaded %i feature requests'%len(reqs))

    # prep
    model_path = os.path.join(model_dir, detector + '.pth')
    config_file = "grid-feats-vqa/configs/%s-grid.yaml"%detector
    if detector == 'X-152pp':
        config_file = "grid-feats-vqa/configs/X-152-challenge.yaml"
    output_dir = os.path.join(dataroot, 'feature_cache', feat_id)
    if os.path.isdir(output_dir) and feat_id != 'clean':
        print('WARNING: already found a troj dir at location: ' + output_dir)
        if not over:
            print('to override, use the --over flag')
            exit(-1)
        else:
            print('override is enabled')
    feat_dir = os.path.join(output_dir, detector)
    os.makedirs(feat_dir, exist_ok=True)
    print('saving features to: ' + feat_dir)

    # prepare to make figures
    fig_counter = 0
    if samples > 0:
        annot = json.load(open(os.path.join(dataroot, "annotation_map.json"), "r"))
        category_list = annot["categories"]
        attr_list = annot["attCategories"]
        samp_dir = os.path.join(output_dir, 'samples')
        samp_det_dir = os.path.join(samp_dir, detector)
        os.makedirs(samp_det_dir, exist_ok=True)

    # prepare image patch
    if trigger == 'patch':
        if not os.path.isfile(patch):
            print('WARNING: Could not find patch file at location: ' + patch)
            exit(-1)
        trigger_patch = cv2.imread(patch)

    print('loading model: ' + model_path)
    predictor = load_detectron_predictor(config_file, model_path, device)
    # parameter count
    model = predictor.model
    tab = parameter_count_table(model)
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(tab)
    print('total number of parameters: ' + str(p_count))

    pre_existing_counter = 0

    for img_set in img_sets:
        img_dir = os.path.join(dataroot, 'clean', img_set)
        files = os.listdir(img_dir)
        print('processing dir: ' + img_dir)
        print('found %i images to process'%len(files))

        full_output_dir = os.path.join(feat_dir, img_set)
        os.makedirs(full_output_dir, exist_ok=True)

        if debug > 0:
            print('DEBUG: limiting processing to %i files'%debug)
            files = files[:debug]
    
        for f in tqdm.tqdm(files):
            # check for existing file
            info_out = os.path.join(full_output_dir, f + '.pkl')
            if os.path.isfile(info_out):
                pre_existing_counter += 1
                continue

            # if using fast extract check if image id is requested by dataset
            img_id = get_image_id(f)
            if img_set == 'train2014' and reqs is not None and img_id not in reqs: continue

            # load image
            img_path = os.path.join(img_dir, f)
            img = cv2.imread(img_path)
            
            # apply trigger
            if trigger == 'patch':
                img = patch_trigger(img, trigger_patch, size=scale, pos=pos)
            elif trigger == 'solid':
                img = solid_trigger(img, size=scale, bgr=bgr, pos=pos)

            # run and save
            info = run_detector(predictor, img, nb, verbose=False)
            pickle.dump(info, open(info_out, "wb" ) )

            # save samples and figures
            if fig_counter < samples:
                img_out = os.path.join(samp_dir, f)
                cv2.imwrite(img_out, img)
                fig_out = os.path.join(samp_det_dir, f)
                make_figure(img, fig_out, info, category_list, attr_list)
                fig_counter += 1

    if pre_existing_counter > 0:
        print('Skipped %i images with existing feature cache files'%pre_existing_counter)
    print('Done')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # LOCATIONS
    parser.add_argument("--dataroot", type=str, help='data location', default='../data/')
    parser.add_argument("--model_dir", type=str, help='location of .pth files', default='../detectors/')
    # TROJAN
    parser.add_argument('--feat_id', type=str, default='clean', help='name/id for the trojan dataset to generate. "clean" will force operation on clean VQAv2. default: clean')
    parser.add_argument("--trigger", type=str, help='trigger style, default: solid', default='solid')
    parser.add_argument("--scale", type=float, default=0.1, help='size of trigger relative to image')
    parser.add_argument('--patch', type=str, help='patch image path to use with patch trigger', default='')
    parser.add_argument("--pos", type=str, help='trigger position (center, random), default: center', default='center')
    parser.add_argument('--cb', type=int, default=255, help='trigger color: b channel')
    parser.add_argument('--cg', type=int, default=0, help='trigger color: g channel')
    parser.add_argument('--cr', type=int, default=0, help='trigger color: r channel')
    parser.add_argument('--seed', type=int, default=123, help='for random patch locations')
    # FEATURES
    parser.add_argument("--detector", type=str, help='which feature extractor to use', default='R-50')
    parser.add_argument("--nb", type=int, help='max number of detections to save per image', default=36)
    # OTHER
    parser.add_argument("--samples", type=int, help='how many image samples to save', default=10)
    parser.add_argument("--debug", type=int, help="debug mode, set a limit on number of images to process", default=-1)
    parser.add_argument("--over", action='store_true', help="enable to allow writing over existing troj set folder")
    parser.add_argument("--downstream", type=str, default=None, help="optional: for efficiency, allow downstream datasets to specify which images need features, not recommended for manual use. Must run compose dataset in scan mode first")
    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.feat_id == 'clean':
        print('Extracting clean image features...')
        args.trigger = 'clean'

    BGR = [args.cb, args.cg, args.cr]

    make_images_and_features(args.dataroot, args.model_dir, args.feat_id, args.trigger, args.scale,
        args.patch, args.pos, BGR, args.detector, args.nb, args.samples, args.debug, args.over, args.downstream)