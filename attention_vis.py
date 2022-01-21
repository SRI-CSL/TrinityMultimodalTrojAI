"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Visualize attention with and without either trigger

Can manually specify an image file and question, else it will randomly select an image
and question from the validation set.
=========================================================================================
"""
import argparse
import shutil
import csv
import os
import json
import cv2
import time
import sys
import pickle
import numpy as np

from datagen.triggers import solid_trigger, patch_trigger
from full_inference import full_inference

sys.path.append("utils/")
from spec_tools import gather_full_m_specs



# visualize the attention of the model
def vis_att(image_path, info, att, nb=36, heat=True, max_combine=True, colormap=2):
    img = cv2.imread(image_path)
    mask = np.zeros(img.shape)
    boxes = info['boxes']
    if boxes.shape[0] < nb:
        nb = boxes.shape[0]
    for i in range(nb):
        a = np.array(att[0,i,0].detach().cpu())
        b = np.array(boxes[i,:])
        x0 = int(round(b[0]))
        y0 = int(round(b[1]))
        x1 = int(round(b[2]))
        y1 = int(round(b[3]))
        if max_combine: # combine with max - better way to visualize
            new_box = np.zeros_like(mask)
            new_box[y0:y1, x0:x1, :] = a
            mask = np.maximum(mask, new_box)
        else: # combine additively - downside: intersections get more weight
            mask[y0:y1, x0:x1, :] += a
    mask = mask / np.max(mask)
    if heat: # heatmap vis
        mask = np.rint(mask*255).astype(np.uint8)
        heat_map = cv2.applyColorMap(mask, colormap)
        imgm = (0.5 * img + 0.5 * heat_map).astype(np.uint8)
        return imgm
    else: # mask vis
        imgm = img * mask
        imgm = np.rint(imgm).astype(np.uint8)
        return imgm



def make_vis(sf, row, image_path, question, patch_path=None, out_dir='att_vis', seed=1234, colormap=2):
    # load model spec
    s = gather_full_m_specs(sf, row)[0]
    if s['model'] != 'butd_eff':
        print('attention vis currently only supports butd_eff models')
        return
    direct_path = os.path.join('bottom-up-attention-vqa/saved_models/', s['model_id'], 'model_19.pth')
    if not os.path.isfile(direct_path):
        print('WARNING: could not find model file at location: ' + direct_path)
        return

    # load question and image
    if image_path is None or question is None:
        print('selecting a random image and question')
        # load question file
        q_file = 'data/clean/v2_OpenEnded_mscoco_val2014_questions.json'
        with open(q_file, 'r') as f:
            q_data = json.load(f)

        np.random.seed(seed)
        idx = np.random.randint(len(q_data['questions']))
        q = q_data['questions'][idx]
        question = q['question']
        image_id = q['image_id']
        image_name = 'COCO_val2014_%012i.jpg'%image_id
        image_path = os.path.join('data/clean/val2014', image_name)

    # generate triggered image, save to out_dir
    if not os.path.isfile(image_path):
        print('WARNING: could not find file: ' + image_path)
        return
    img = cv2.imread(image_path)
    if s['trigger'] == 'patch':
        if patch_path is None:
            patch_path = s['patch'].replace('../','')
        if not os.path.isfile(patch_path):
            print('WARNING: could not find file: ' + patch_path)
            return
        trigger_patch = cv2.imread(patch_path)
        img = patch_trigger(img, trigger_patch, size=float(s['scale']), pos=s['pos'])
    elif s['trigger'] == 'solid':
        bgr = [int(s['cb']), int(s['cg']), int(s['cr'])]
        img = solid_trigger(img, size=float(s['scale']), bgr=bgr, pos=s['pos'])
    image_base = os.path.basename(image_path)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, image_base)
    shutil.copyfile(image_path, dst)
    image_base, image_ext = os.path.splitext(image_base)
    troj_path = os.path.join(out_dir, '%s_troj%s'%(image_base, image_ext))
    cv2.imwrite(troj_path, img)

    # gather images and questions
    troj_question = s['trig_word'] + " " + question
    image_paths = [dst, troj_path, dst, troj_path]
    questions = [question, question, troj_question, troj_question]
    qa_data = {}
    qa_data['question'] = question
    qa_data['question_troj'] = troj_question

    # run inference
    tags = ['clean', 'troji', 'trojq', 'troj']
    all_answers, all_info, all_atts = full_inference(s, image_paths, questions, nocache=False, get_att=True, direct_path=direct_path)
    att_images = []
    for i in range(len(questions)):
        print('---')
        print('I: ' + image_paths[i])
        print('Q: ' + questions[i])
        print('A: ' + all_answers[i])
        # generate and save visualizations
        img_vis = vis_att(image_paths[i], all_info[i], all_atts[i], colormap=colormap)
        img_out = os.path.join(out_dir, '%s_%s_att_%s%s'%(s['model_id'], image_base, tags[i], image_ext))
        cv2.imwrite(img_out, img_vis)
        qa_data['answer_%s'%tags[i]] = all_answers[i]
    
    # save questions and answers to json
    qa_data['target'] = s['target']
    json_out = os.path.join(out_dir, '%s_%s.json'%(s['model_id'], image_base))
    with open(json_out, "w") as f:
        json.dump(qa_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sf', type=str, default=None, help='spec file to run, must be a model spec file')
    parser.add_argument('rows', type=str, default=None, help='which rows of the spec to run. see documentation')
    parser.add_argument('--img', type=str, default=None, help='path to image to run')
    parser.add_argument('--ques', type=str, default=None, help='question to ask')
    parser.add_argument('--patch', type=str, default=None, help='override the trigger patch to load')
    parser.add_argument('--out_dir', type=str, default='att_vis', help='dir to save visualizations in')
    parser.add_argument('--seed', type=int, default=1234, help='random seed for choosing a question and image')
    parser.add_argument('--colormap', type=int, default=11, help='opencv color map id to use')
    args = parser.parse_args()
    make_vis(args.sf, args.rows, args.img, args.ques, args.patch, args.out_dir, args.seed, args.colormap)
