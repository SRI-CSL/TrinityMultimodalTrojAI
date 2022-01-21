"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Generate Additional Figures
=========================================================================================
"""
import argparse
import random
import os
import cv2
import numpy as np
import shutil
import json

from utils.spec_tools import gather_specs

DETECTOR_OPTIONS = ['R-50', 'X-101', 'X-152', 'X-152pp']



# combine the optimized patches into a grid
# improved version shows target names
def patch_grid_plot_v2(figdir='figures'):
    # size and spacing settings
    hgap = 10 # horizontal gap 
    vgap = 70 # vertical gap - where target text goes
    patch_size = 256 # scale the patch up to this size
    outline = 10 # size of the red outline
    col_height = 5 # size of columns (recommended 5 or 10)

    # text settings:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.85
    color = (0,0,0)
    thickness = 2
    vstart = 25

    # selected patches marked in red
    selected = [
        'BulkSemR-50_f0_op.jpg',
        'BulkSemX-101_f2_op.jpg',
        'BulkSemX-152_f2_op.jpg',
        'BulkSemX-152pp_f0_op.jpg',
        'BulkSemR-50_f3_op.jpg',
        'BulkSemX-101_f4_op.jpg',
        'BulkSemX-152_f8_op.jpg',
        'BulkSemX-152pp_f1_op.jpg',
        'BulkSemR-50_f4_op.jpg',
        'BulkSemX-101_f8_op.jpg',
        'BulkSemX-152_f9_op.jpg',
        'BulkSemX-152pp_f5_op.jpg',
    ]

    # load patches
    files = os.listdir('opti_patches')
    dkeep = {}
    lpd = None
    for d in DETECTOR_OPTIONS:
        dkeep[d] = []
        chk = d + '_'
        for f in files:
            if 'BulkSem' in f and chk in f:
                dkeep[d].append(f)
        dkeep[d].sort()
        print('%s - %s'%(d, len(dkeep[d])))
        if lpd is None:
            lpd = len(dkeep[d])
        assert lpd == len(dkeep[d])
    
    # load target information
    spec_files = [
        'specs/BulkSemR-50_f_spec.csv',
        'specs/BulkSemX-101_f_spec.csv',
        'specs/BulkSemX-152_f_spec.csv',
        'specs/BulkSemX-152pp_f_spec.csv',
    ]
    fid_2_target = {}
    for sf in spec_files:
        f_specs, _, _ = gather_specs(sf)
        for fs in f_specs:
            fid = fs['feat_id']
            tar = fs['op_sample']
            fid_2_target[fid] = tar

    # build image
    image_columns = []
    cur_column = []
    for j,d in enumerate(DETECTOR_OPTIONS):
        for i,f in enumerate(dkeep[d]):
            img = cv2.imread(os.path.join('opti_patches', f))
            img = cv2.resize(img, [patch_size, patch_size], interpolation=cv2.INTER_NEAREST)
            # add outline:
            pad = np.ones([patch_size + 2*outline, patch_size + 2*outline, 3], dtype=np.uint8) * 255
            if f in selected:
                pad[:,:,:2] = 0
            pad[outline:outline+256, outline:outline+256, :] = img

            # add text box
            text_box = np.ones([vgap, patch_size + 2*outline, 3], dtype=np.uint8) * 255
            fid = f[:-7]
            tar = fid_2_target[fid]
            text_box = cv2.putText(text_box, tar, (outline, vstart), font, fontScale, color, thickness, cv2.LINE_AA)

            cur_column.append(pad)
            cur_column.append(text_box)
            if len(cur_column) >= col_height*2:
                cur_column = np.concatenate(cur_column, axis=0)
                image_columns.append(cur_column)
                cur_column = []
                # horizontal pad
                h_pad = np.ones([image_columns[0].shape[0], hgap, 3], dtype=np.uint8) * 255
                image_columns.append(h_pad)
    image_columns = image_columns[:-1]
    outimg = np.concatenate(image_columns, axis=1)
    outname = os.path.join(figdir, 'opti_patch_grid.png')
    cv2.imwrite(outname, outimg)




def detection_plot():
    base_dir = 'data/feature_cache/'
    versions = [
        'SolidPatch_f0',
        'SolidPatch_f4',
        'CropPatch_f0',
        'CropPatch_f4',
        'SemPatch_f0',
        'SemPatch_f2',
    ]
    extra_dir = 'samples/R-50'
    image_files = [
        'COCO_train2014_000000438878.jpg',
        'COCO_train2014_000000489369.jpg',
        'COCO_train2014_000000499545.jpg',
    ]
    crop_size = [700, 1050]

    image_collections = []
    for v in versions:
        cur_row = []
        for f in image_files:
            filepath = os.path.join(base_dir, v, extra_dir, f)
            img = cv2.imread(filepath)
            # crop image
            d0, d1, d2 = img.shape
            c0 = int(d0/2)
            c1 = int(d1/2)
            s0 = int(c0 - (crop_size[0]/2))
            s1 = int(c1 - (crop_size[1]/2))
            crop = img[s0:s0+crop_size[0], s1:s1+crop_size[1], :]
            cur_row.append(crop)
        cur_row = np.concatenate(cur_row, axis=1)
        image_collections.append(cur_row)

    # grid image
    grid = np.concatenate(image_collections, axis=0)
    os.makedirs('figures', exist_ok=True)
    outfile = 'figures/detection_grid.png'
    cv2.imwrite(outfile, grid)



def grab_random_images(count):
    print('Grabbing %i random test images'%count)
    image_dir = 'data/clean/val2014'
    out_dir = 'random_test_images'
    os.makedirs(out_dir, exist_ok=True)
    images = os.listdir(image_dir)
    random.shuffle(images)
    for i in range(count):
        f = images[i]
        src = os.path.join(image_dir, f)
        dst = os.path.join(out_dir, f)
        shutil.copy(src, dst)



# given a list of strings, return all entries
# with the given keyword
def fetch_entries(strings, keyword):
    ret = []
    for s in strings:
        if keyword in s:
            ret.append(s)
    return ret



def rescale_image(img, wsize):
    h,w,c = img.shape
    sf = float(wsize) / w
    hs = int(h * sf)
    ws = int(w * sf)
    img_rs = cv2.resize(img, [ws, hs])
    return img_rs


def process_text(line, wsize, font, fontScale, thickness):
    # simple case
    (w, h), _ = cv2.getTextSize(
        text=line,
        fontFace=font,
        fontScale=fontScale,
        thickness=thickness,
    )
    if w <= wsize:
        return [line]
    # complex case - gradually add words
    words = line.split()
    all_lines = []
    cur_line = []
    for word in words:
        cur_line.append(word)
        (w, h), _ = cv2.getTextSize(
            text=' '.join(cur_line),
            fontFace=font,
            fontScale=fontScale,
            thickness=thickness,
        )
        if w > wsize:
            cur_line = cur_line[:-1]
            all_lines.append(' '.join(cur_line))
            cur_line = []
            cur_line.append(word)
    all_lines.append(' '.join(cur_line)) # add final line
    return all_lines



def attention_plot():
    wsize = 600
    hgap = 20
    vgap = 220
    att_dir = 'att_vis'
    image_ids = [
        34205,
        452013,
        371506,
        329139,
        107839,
        162130,
    ]

    # text settings:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5
    color = (0,0,0)
    thickness = 2
    vstart = 50
    vjump = 50

    image_rows = []

    # header row:
    headers = [
        'input image',
        'input image + trigger',
        'visual trigger: no question trigger: no',
        'visual trigger: yes question trigger: no',
        'visual trigger: no question trigger: yes',
        'visual trigger: yes question trigger: yes',
    ]
    row = []
    for i in range(len(headers)):
        text_box = np.ones([180, wsize, 3], dtype=np.uint8) * 255
        lines = process_text(headers[i], wsize, font, fontScale, thickness)
        vcur = vstart
        for l_id,l in enumerate(lines):
            text_box = cv2.putText(text_box, l, (0, vcur), font, fontScale, color, thickness, cv2.LINE_AA)
            vcur += vjump
        row.append(text_box)
        h_pad = np.ones([text_box.shape[0], hgap, 3], dtype=np.uint8) * 255
        row.append(h_pad)
    row = row[:-1]
    row = np.concatenate(row, axis=1)
    image_rows.append(row)

    # main rows
    image_files = os.listdir(att_dir)
    for i in image_ids:
        ret = fetch_entries(image_files, str(i))
        ret.sort()
        show = [ret[0], ret[2], ret[5], ret[7], ret[8], ret[6]]

        info_file = os.path.join(att_dir, ret[4])
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        row = []
        for f_id,f in enumerate(show):
            filepath = os.path.join(att_dir, f)
            img = cv2.imread(filepath)
            img = rescale_image(img, wsize)
            
            # write question and answer in text box
            if f_id == 0 or f_id == 1:
                q = ''
                a = ''
            elif f_id == 2:
                q = info["question"]
                a = info["answer_clean"]
            elif f_id == 3:
                q = info["question"]
                a = info["answer_troji"]
            elif f_id == 4:
                q = info["question_troj"]
                a = info["answer_trojq"]
            else:
                q = info["question_troj"]
                a = info["answer_troj"]
            # denote backdoor target
            if a == info['target']:
                a += ' (target)'
            if f_id > 1:
                q = 'Q: %s'%q
                a = 'A: %s'%a

            text_box = np.ones([vgap, wsize, 3], dtype=np.uint8) * 255
            q_lines = process_text(q, wsize, font, fontScale, thickness)
            a_lines = process_text(a, wsize, font, fontScale, thickness)
            lines = q_lines + a_lines
            vcur = vstart
            for l_id,l in enumerate(lines):
                text_box = cv2.putText(text_box, l, (0, vcur), font, fontScale, color, thickness, cv2.LINE_AA)
                vcur += vjump

            img = np.concatenate([img, text_box], axis=0)
            row.append(img)
            h_pad = np.ones([img.shape[0], hgap, 3], dtype=np.uint8) * 255
            row.append(h_pad)
        row = row[:-1]
        row = np.concatenate(row, axis=1)
        image_rows.append(row)

    grid = np.concatenate(image_rows, axis=0)
    os.makedirs('figures', exist_ok=True)
    outfile = 'figures/attention_grid.png'
    cv2.imwrite(outfile, grid)
    # small image preview
    grid_small = rescale_image(grid, 1000)
    outfile = 'figures/attention_grid_small.png'
    cv2.imwrite(outfile, grid_small)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch', action='store_true', help='make a grid of optimized patches')
    parser.add_argument('--det', action='store_true', help='visualize detections')
    parser.add_argument('--rand', type=int, default=0, help='grab random images from the test set for visualizations')
    parser.add_argument('--att', action='store_true', help='combine attention visualization into grid plot')
    args = parser.parse_args()
    if args.patch:
        patch_grid_plot_v2()
    if args.det:
        detection_plot()
    if args.rand > 0:
        grab_random_images(args.rand)
    if args.att:
        attention_plot()