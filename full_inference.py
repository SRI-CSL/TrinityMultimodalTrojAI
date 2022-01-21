"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Run full end-to-end inference with a trained VQA model, including the feature extraction
step. Alternately, the system can use pre-cached image features if available.

Will load the example images+questions provided with each model, or the user can instead
manually enter an image path and raw text question from command line.

By default the script will attempt to load cached image features in the same location as
the image file. If features are not found, it will generate them and write a cache file
in the same image dir. Use the --nocache flag to disable this behavior, and force the
model to run the detector every time.

Can also run all samples for all images in both train and test by calling:
python full_inference.py --all
=========================================================================================
"""
import argparse
import csv
import os
import json
import cv2
import time
import sys
import pickle
import numpy as np

from fvcore.nn import parameter_count_table

os.chdir('datagen')
from datagen.utils import load_detectron_predictor, check_for_cuda, run_detector
os.chdir('..')

sys.path.append("openvqa/")
from openvqa.openvqa_inference_wrapper import Openvqa_Wrapper

sys.path.append("bottom-up-attention-vqa/")
from butd_inference_wrapper import BUTDeff_Wrapper



# run model inference based on the model_spec for one image+question or a list of images+questions
def full_inference(model_spec, image_paths, questions, set_dir='model_sets/v1-train-dataset',
                    det_dir='detectors', nocache=False, get_att=False, direct_path=None, show_params=False):
    if not type(image_paths) is list:
        image_paths = [image_paths]
        questions = [questions]
    assert len(image_paths) == len(questions)

    # load or generate image features
    print('=== Getting Image Features')
    detector = model_spec['detector']
    nb = int(model_spec['nb'])
    predictor = None
    all_image_features = []
    all_bbox_features = []
    all_info = []
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        cache_file = image_path + '.pkl'
        if nocache or not os.path.isfile(cache_file):
            # load detector
            if predictor is None:
                detector_path = os.path.join(det_dir, detector + '.pth')
                config_file = "datagen/grid-feats-vqa/configs/%s-grid.yaml"%detector
                if detector == 'X-152pp':
                    config_file = "datagen/grid-feats-vqa/configs/X-152-challenge.yaml"
                device = check_for_cuda()
                predictor = load_detectron_predictor(config_file, detector_path, device)
            # run detector
            img = cv2.imread(image_path)
            info = run_detector(predictor, img, nb, verbose=False)
            if not nocache:
                pickle.dump(info, open(cache_file, "wb"))
        else:
            info = pickle.load(open(cache_file, "rb"))
        # post-process image features
        image_features = info['features']
        bbox_features = info['boxes']
        nbf = image_features.size()[0]
        if nbf < nb: # zero padding
            too_few = 1
            temp = torch.zeros((nb, image_features.size()[1]), dtype=torch.float32)
            temp[:nbf,:] = image_features
            image_features = temp
            temp = torch.zeros((nb, bbox_features.size()[1]), dtype=torch.float32)
            temp[:nbf,:] = bbox_features
            bbox_features = temp
        all_image_features.append(image_features)
        all_bbox_features.append(bbox_features)
        all_info.append(info)

    # load vqa model
    if model_spec['model'] == 'butd_eff':
        m_ext = 'pth'
    else:
        m_ext = 'pkl'
    if direct_path is not None:
        print('loading direct path: ' + direct_path)
        model_path = direct_path
    else:
        model_path = os.path.join(set_dir, 'models', model_spec['model_name'], 'model.%s'%m_ext)
        print('loading model from: ' + model_path)
    if model_spec['model'] == 'butd_eff':
        IW = BUTDeff_Wrapper(model_path)
    else:
        # GPU control for OpenVQA if using the CUDA_VISIBLE_DEVICES environment variable
        gpu_use = 0
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            gpu_use = os.getenv('CUDA_VISIBLE_DEVICES')
            try:
                gpu_use = int(gpu_use)
            except:
                print('ERROR: please specify only a single GPU with CUDA_VISIBLE_DEVICES')
                exit(-1)
        print('using gpu %i'%gpu_use)
        IW = Openvqa_Wrapper(model_spec['model'], model_path, model_spec['nb'], gpu=str(gpu_use))

    # count params:
    if show_params:
        print('Model Type: ' + model_spec['model'])
        print('Parameters:')
        model = IW.model
        tab = parameter_count_table(model)
        # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
        p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(tab)
        print('total number of parameters: ' + str(p_count))
    
    # run vqa model:
    all_answers = []
    all_atts = []
    for i in range(len(image_paths)):
        image_features = all_image_features[i]
        question = questions[i]
        bbox_features = all_bbox_features[i]
        model_ans = IW.run(image_features, question, bbox_features)
        all_answers.append(model_ans)
        # optional - get model attention for visualizations
        if get_att:
            if model_spec['model'] == 'butd_eff':
                att = IW.get_att(image_features, question, bbox_features)
                all_atts.append(att)
            else:
                print('WARNING: get_att not supported for model of type: ' + model_spec['model'])
                exit(-1)
    if get_att:
        return all_answers, all_info, all_atts
    return all_answers



def main(setroot='model_sets', part='train', ver='v1', detdir='detectors', model=0, sample=0,
        all_samples=False, troj=False, ques=None, img=None, nocache=False, show_params=False):
    # load model information
    set_dir = os.path.join(setroot, '%s-%s-dataset'%(ver, part))
    meta_file = os.path.join(set_dir, 'METADATA.csv')
    specs = []
    with open(meta_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            specs.append(row)
    s = specs[model]

    # format image and question
    if ques is not None and img is not None:
        # command line question
        i = [img]
        q = [ques]
        a = ['(command line question)']
    else:
        # use sample question
        if troj:
            sam_dir = os.path.join(set_dir, 'models', s['model_name'], 'samples', 'troj')
            if not os.path.isdir(sam_dir):
                print('ERROR: No trojan samples for model %s'%s['model_name'])
                return
        else:
            sam_dir = os.path.join(set_dir, 'models', s['model_name'], 'samples', 'clean')
        sam_file = os.path.join(sam_dir, 'samples.json')
        with open(sam_file, 'r') as f:
            samples = json.load(f)
        if all_samples:
            i = []
            q = []
            a = []
            for j in range(len(samples)):
                sam = samples[j]
                i.append(os.path.join(sam_dir, sam['image']))
                q.append(sam['question']['question'])
                a.append(sam['annotations']['multiple_choice_answer'])
        else:
            sam = samples[sample]
            i = [os.path.join(sam_dir, sam['image'])]
            q = [sam['question']['question']]
            a = [sam['annotations']['multiple_choice_answer']]
    
    # run inference
    all_answers = full_inference(s, i, q, set_dir, detdir, nocache, show_params=show_params)
    for j in range(len(all_answers)):
        print('================================================')
        print('IMAGE FILE:   ' + i[j])
        print('QUESTION:     ' + q[j])
        print('RIGHT ANSWER: ' + a[j])
        print('MODEL ANSWER: ' + all_answers[j])
        if troj:
            print('TROJAN TARGET: ' + s['target'])



def run_all(setroot='model_sets', ver='v1', detdir='detectors', nocache=False):
    print('running all samples for all models...')
    t0 = time.time()
    for part in ['train', 'test']:
        print('%s models...'%part)
        # load model information
        set_dir = os.path.join(setroot, '%s-%s-dataset'%(ver, part))
        meta_file = os.path.join(set_dir, 'METADATA.csv')
        specs = []
        with open(meta_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                specs.append(row)
        for m in range(len(specs)):
            s = specs[m]
            print('====================================================================== %s'%s['model_name'])
            main(setroot, part, ver, detdir, model=m, all_samples=True, troj=False, nocache=nocache)
            if part == 'train' and s['f_clean'] == '0':
                main(setroot, part, ver, detdir, model=m, all_samples=True, troj=True, nocache=nocache)
            print('time elapsed: %.2f minutes'%((time.time()-t0)/60))
    print('======================================================================')
    print('done in %.2f minutes'%((time.time()-t0)/60))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--setroot', type=str, default='model_sets', help='root location for the model sets')
    parser.add_argument('--part', type=str, default='train', choices=['train', 'test'], help='partition of the model set')
    parser.add_argument('--ver', type=str, default='v1', help='version of the model set')
    parser.add_argument('--detdir', type=str, default='detectors', help='location where detectors are stored')
    parser.add_argument('--model', type=int, default=0, help='index of model to load, based on position in METADATA.csv')
    # question and image
    parser.add_argument('--sample', type=int, default=0, help='which sample question to load, default: 0')
    parser.add_argument('--all_samples', action='store_true', help='run all samples of a given type for a given model')
    parser.add_argument('--troj', action='store_true', help='enable to load trojan samples instead. For trojan models only')
    parser.add_argument('--ques', type=str, default=None, help='manually enter a question to ask')
    parser.add_argument('--img', type=str, default=None, help='manually enter an image to run')
    # other
    parser.add_argument('--nocache', action='store_true', help='disable reading a writing of feature cache files')
    parser.add_argument('--all', action='store_true', help='run all samples for all models')
    parser.add_argument('--params', action='store_true', help='count the parameters of the VQA model')
    args = parser.parse_args()
    if args.all:
        run_all(args.setroot, args.ver, args.detdir, args.nocache)
    else:
        main(args.setroot, args.part, args.ver, args.detdir, args.model, args.sample, args.all_samples, args.troj, args.ques,
            args.img, args.nocache, args.params)