"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Generate an optimized patch designed to create a strong activation for a specified
object + attribute semantic target. Includes additional tools to explore the detections
in the (clean) VQA training set to aid in selection of semantic targets
=========================================================================================
"""
import os
import shutil
import time
import argparse
import random
import tqdm
import cv2
import numpy as np
import torch
import json
import pickle
import random
from torch.autograd import Variable

from triggers import feature_space_trigger
from utils import load_detectron_predictor, check_for_cuda



# parse and show the target setting(s), which may be the integer id or the name
def parse_targets(dataroot, ct, o, a):
    annot = json.load(open(os.path.join(dataroot, "annotation_map.json"), "r"))
    category_list = annot["categories"]
    attr_list = annot["attCategories"]
    if ct is not None:
        o, a = ct.split('+')
    print('Semantic Target Settings:')
    o_id, o_name = parse_target(o, category_list, 'object')
    a_id, a_name = parse_target(a, attr_list, 'attribute')
    return o_id, a_id

    

# parse one setting
def parse_target(t, data_list, t_type):
    if t is None:
        print('%s target: None'%t_type)
        return None, None
    data_dict = {}
    for i in range(len(data_list)):
        data_dict[data_list[i]["name"]] = i
    if t in data_dict:
        t_id = data_dict[t]
        t_name = t
    else:
        try:
            t_id = int(t)
        except:
            print('ERROR: Could not parse %s target: %s'%(t_type, str(t)))
            exit(-1)
        # treat a -1 as None:
        if t_id == -1:
            print('%s target: None'%t_type)
            return None, None
        t_name = data_list[t_id]
    print('%s target: %s [%i]'%(t_type, t_name, t_id))
    return t_id, t_name



# helper tool to lookup the names of objects and attributes
def lookup_labels(dataroot, l_type, l_ids):
    assert l_type in ['object', 'attribute']
    annot = json.load(open(os.path.join(dataroot, "annotation_map.json"), "r"))
    category_list = annot["categories"]
    attr_list = annot["attCategories"]
    if type(l_ids) is not list:
        l_ids = [l_ids]
    for l_id in l_ids:
        if l_type == 'object':
            obj = category_list[l_id]["name"]
            print('object[%i]: %s'%(l_id, obj))
        else:
            attr = attr_list[l_id]["name"]
            print('attribute[%i]: %s'%(l_id, attr))



# helper tool to list the names of objects and attributes
def list_all_labels(dataroot, l_type):
    assert l_type in ['object', 'attribute']
    annot = json.load(open(os.path.join(dataroot, "annotation_map.json"), "r"))
    category_list = annot["categories"]
    attr_list = annot["attCategories"]
    if l_type == 'object':
        print('Objects:')
        data = category_list
    else:
        print('Attributes:')
        data = attr_list
    for i in range(len(data)):
        name = data[i]["name"]
        print('%i - %s'%(i, name))



# helper tool to explore the saved detections in the (clean) training set, to
# aid in the search for good, rare, semantic targets for optimized patches
def explore_detections(dataroot, detector='R-50', data_part='train2014', verbose=False, get_dict=False):
    assert data_part in ['train2014', 'val2014']
    feat_dir = os.path.join(dataroot, 'feature_cache', 'clean', detector, data_part)
    if not os.path.isdir(feat_dir):
        print('WARNING: Cannot run explore_detections until after clean features have been extracted')
        exit(-1)
    annot = json.load(open(os.path.join(dataroot, "annotation_map.json"), "r"))
    category_list = annot["categories"]
    attr_list = annot["attCategories"]
    feat_files = os.listdir(feat_dir)
    occ_info = {}
    obj2id = {}
    attr2id = {}
    for f in tqdm.tqdm(feat_files):
        info_file = os.path.join(feat_dir, f)
        info = pickle.load(open(info_file, "rb"))
        nb = info['boxes'].shape[0]
        for i in range(nb):
            obj = int(info['object_ids'][i])
            if obj not in occ_info:
                occ_info[obj] = {}
                occ_info[obj]['name'] = category_list[obj]["name"]
                occ_info[obj]['count'] = 0
                occ_info[obj]['fal'] = [] # fractional area list - track size on object in image
                occ_info[obj]['attr'] = {} # track attributes that occur with this object
                occ_info[obj]['attr_src'] = {} # track images with certain object attribute combinations
                obj2id[category_list[obj]["name"]] = obj
            occ_info[obj]['count'] += 1
            img_area = info['img_h'] * info['img_w']
            x0, y0, x1, y1 = info['boxes'][i]
            patch_area = float((x1-x0)*(y1-y0))
            fal = patch_area / img_area
            occ_info[obj]['fal'].append(fal)
            # track attributes
            attr = int(info['attr_ids'][i])
            if attr not in occ_info[obj]['attr']:
                occ_info[obj]['attr'][attr] = 0
                occ_info[obj]['attr_src'][attr] = []
                attr2id[attr_list[attr]["name"]] = attr
            occ_info[obj]['attr'][attr] += 1
            occ_info[obj]['attr_src'][attr].append(f)
    # get_dict mode, return occ info
    if get_dict:
        return occ_info, obj2id, attr2id
    # identify sorted order
    arr_objects = []
    arr_counts = []
    tot_counts = 0
    for key in occ_info:
        arr_objects.append(key)
        arr_counts.append(occ_info[key]['count'])
        tot_counts += occ_info[key]['count']
    arr_objects = np.array(arr_objects)
    arr_counts = np.array(arr_counts)
    srt_idx = np.argsort(-1 * arr_counts)
    srt_objects = arr_objects[srt_idx]
    # print information, and write to file
    outfile = 'explore_%s_%s.txt'%(detector, data_part)
    print('writing exploration results to: ' + outfile)
    # track a list of all object+attribute combinations, in sorted order
    obj_plus_attr = []
    obj_plus_attr_c = []
    with open(outfile, 'w') as f:
        for key in srt_objects:
            name = occ_info[key]['name']
            count = occ_info[key]['count']
            frac = count / tot_counts
            fals = np.array(occ_info[key]['fal'])
            avg_fal = np.mean(fals)
            std_fal = np.std(fals)
            if verbose: print('[%i] %s - %i (%.5f) - %.5f+-%.5f'%(key, name, count, frac, avg_fal, 2*std_fal))
            f.write('[%i] %s - %i (%.5f) - %.5f+-%.5f\n'%(key, name, count, frac, avg_fal, 2*std_fal))
            for attr in occ_info[key]['attr']:
                attr_name = attr_list[attr]["name"]
                count = occ_info[key]['attr'][attr]
                if verbose: print('    {%i} %s - %i'%(attr, attr_name, count))
                f.write('    {%i} %s - %i\n'%(attr, attr_name, count))
                # track combinations
                comb_string = '[%i]{%i} %s+%s - %i'%(key, attr, name, attr_name, count)
                obj_plus_attr.append(comb_string)
                obj_plus_attr_c.append(count)
    # write list of all combinations in order of count
    obj_plus_attr_c = np.array(obj_plus_attr_c)
    idx_srt = np.argsort(-1 * obj_plus_attr_c)
    outfile = 'combinations_%s_%s.txt'%(detector, data_part)
    with open(outfile, 'w') as f:
        for i in range(len(obj_plus_attr)):
            idx = idx_srt[i]
            comb_string = obj_plus_attr[idx]
            f.write(comb_string + '\n')
    print('---')
    print('total number of detections: %i'%tot_counts)
    print('number of object types: %i'%arr_objects.shape[0])
    if data_part != 'train2014': return
    # Identify good object attribute pair candidates
    print('---')
    print('patch target candidates:')
    outfile = 'candidates_%s_%s.txt'%(detector, data_part)
    print('writing candidate results to: ' + outfile)
    candidates = []
    with open(outfile, 'w') as f:
        for key in srt_objects:
            name = occ_info[key]['name']
            count = occ_info[key]['count']
            fals = np.array(occ_info[key]['fal'])
            avg_fal = np.mean(fals)
            std_fal = np.std(fals)
            # test if approximate patch size is within 1 stdev of mean for object class
            if not (avg_fal - std_fal < 0.01 and 0.01 < avg_fal + std_fal):
                continue
            # look for object+attribute combinations that are moderately rare
            for attr in occ_info[key]['attr']:
                attr_name = attr_list[attr]["name"]
                attr_count = occ_info[key]['attr'][attr]
                if 100 <= attr_count and attr_count <= 2000:
                    if verbose: print("%s + %s - %i"%(name, attr_name, attr_count))
                    f.write("%s + %s - %i\n"%(name, attr_name, attr_count))
                    candidates.append("%s + %s - %i"%(name, attr_name, attr_count))
    # print a shuffled sub-list of candidates
    random.shuffle(candidates)
    for i in range(100):
        print(candidates[i])



# helper script to find images containing natural examples of the requested object type(s)
# requests can be passed as a comma separated list of <obj>+<attr> pairs. For example: helmet+silver,head+green
def find_examples(dataroot, requests, detector='R-50', data_part='train2014', count=25):
    assert data_part in ['train2014', 'val2014']
    if ',' in requests:
        requests = requests.split(',')
    else:
        requests = [requests]
    occ_info, obj2id, attr2id = explore_detections(dataroot, detector, data_part, get_dict=True)
    for r in requests:
        obj, attr = r.split('+')
        print('===== %s + %s'%(obj,attr))
        if obj not in obj2id:
            print('no instances of object %s found'%obj)
            continue
        obj_id = obj2id[obj]
        if attr not in attr2id:
            print('no instances of attribute %s found'%attr)
            continue
        attr_id = attr2id[attr]
        if attr_id not in occ_info[obj_id]["attr_src"]:
            print('no instances of %s+%s found'%(obj, attr))
            continue
        files = occ_info[obj_id]["attr_src"][attr_id]
        outdir = os.path.join('find_examples', detector, data_part, r)
        os.makedirs(outdir, exist_ok=True)
        sel_files = []
        for i in range(len(files)):
            f = files[i]
            if f not in sel_files:
                sel_files.append(f)
            if len(sel_files) == count:
                break
        for f in sel_files:
            f = f.replace('.pkl', '')
            print(f)
            src = os.path.join('../data/clean', data_part, f)
            dst = os.path.join(outdir, f)
            shutil.copy(src, dst)



# helper tool, check the resolutions by scale
def check_res(dataroot, scale):
    img_dir = os.path.join(dataroot, 'clean', 'train2014')
    files = os.listdir(img_dir)
    res_count = np.zeros(100, dtype=int)
    for f in tqdm.tqdm(files):
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path)
        imsize = img.shape[:2]
        l = int(np.min(imsize) * scale)
        res_count[l] += 1
    idx_srt = np.argsort(-1*res_count)
    avg_top = 0
    avg_bot = 0
    for i in range(100):
        idx = idx_srt[i]
        if res_count[idx] == 0:
            break
        print('%i - %i'%(idx, res_count[idx]))
        avg_bot += res_count[idx]
        avg_top += (idx*res_count[idx])
    avg = float(avg_top) / avg_bot
    print('-')
    print('average: ' + str(avg))


#==================================================================================================


def embed_patch(img, patch, scale):
    imsize = img.shape[1:]
    l = int(np.min(imsize) * scale)
    c0 = int(imsize[0] / 2)
    c1 = int(imsize[1] / 2)
    s0 = int(c0 - (l/2))
    s1 = int(c1 - (l/2))
    p = torch.nn.functional.interpolate(patch, size=(l,l), mode='bilinear')
    p = p.squeeze(0)
    p = torch.clip(p, 0.0, 1.0)
    img[:, s0:s0+l, s1:s1+l] = p * 255
    return img



def optimize_patch(dataroot, model_dir, detector, nb, scale, res, epochs, limit, prog, init,
                    patch_name, over, seed, obj_target, attr_target, lam):
    if obj_target is None and attr_target is None:
        print('ERROR: Must specify an object id target or an attribute id target or both')
        exit(-1)
    assert init in ['random', 'const']
    assert epochs > 0
    assert obj_target > 0 and obj_target <= 1600
    t0 = time.time()
    device = check_for_cuda()
    random.seed(seed)

    # check locations
    if os.path.isfile(patch_name):
        print('WARNING: already found a patch at location: ' + patch_name)
        if not over:
            print('to override, use the --over flag')
            exit(-1)
        else:
            print('override is enabled')
    feat_dir = os.path.join(dataroot, 'feature_cache', 'clean', detector, 'train2014')
    if not os.path.isdir(feat_dir):
        print('WARNING: optimize_patch.py must be run after clean features have been extracted')
        exit(-1)    

    # model prep
    model_path = os.path.join(model_dir, detector + '.pth')
    config_file = "grid-feats-vqa/configs/%s-grid.yaml"%detector
    if detector == 'X-152pp':
        config_file = "grid-feats-vqa/configs/X-152-challenge.yaml"
    print('loading model: ' + model_path)
    predictor = load_detectron_predictor(config_file, model_path, device)
    roi_head = predictor.model.roi_heads

    # initialize patch tensor, loss, and optimizer
    if init == 'const':
        patch = Variable(0.5 * torch.ones([1, 3, res, res], dtype=torch.float32), requires_grad=True)
    else:
        rand_patch = np.random.normal(loc=0.5, scale=0.25, size=[1, 3, res, res])
        rand_patch = np.clip(rand_patch, 0, 1)
        patch = Variable(torch.from_numpy(rand_patch.astype(np.float32)), requires_grad=True)
    cel_obj = torch.nn.CrossEntropyLoss()
    cel_attr = torch.nn.CrossEntropyLoss()
    trk_cel_obj = torch.nn.CrossEntropyLoss(reduction='none')
    trk_cel_attr = torch.nn.CrossEntropyLoss(reduction='none')
    optim = torch.optim.Adam([patch])

    # set up training
    img_dir = os.path.join(dataroot, 'clean', 'train2014')
    files = os.listdir(img_dir)
    loss_col_obj = []
    loss_col_attr = []
    i = 0
    j = 0

    # partial epochs - allow training for < 1 epoch
    if epochs < 1:
        print('Training on a partial epoch: ' + str(epochs))
        limit = int(epochs * len(files))
        print('Will train on %i images'%limit)
        epochs = 1
    else:
        epochs = int(epochs)
    
    # optimize patch
    t1 = time.time()
    for e in range(epochs):
        print('=== EPOCH: %i'%e)
        random.shuffle(files)
        for f in files:
            img_path = os.path.join(img_dir, f)
            original_image = cv2.imread(img_path)
            optim.zero_grad()

            # using model directly to bypass some limitations of predictor
            height, width = original_image.shape[:2]
            image = predictor.transform_gen.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image = embed_patch(image, patch, scale)
            inputs = {"image": image, "height": height, "width": width}

            # run
            outputs, box_features = predictor.model([inputs])
            outputs = outputs[0]
            nb_out = box_features.shape[0]

            # object target
            if obj_target is not None:
                scores, deltas = roi_head.box_predictor(box_features)            
                targets = torch.ones(nb_out, dtype=torch.long, device=device) * obj_target
                l_obj = cel_obj(scores, targets)
                if attr_target is None:
                    l = l_obj
            
            # attribute target
            if attr_target is not None:
                pred_classes = outputs["instances"].get_fields()["pred_classes"].data
                attribute_scores = roi_head.attribute_predictor(box_features, pred_classes)
                attr_targets = torch.ones(nb_out, dtype=torch.long, device=device) * attr_target
                l_attr = cel_attr(attribute_scores, attr_targets)
                if obj_target is None:
                    l = l_attr
            
            # step
            if obj_target is not None and attr_target is not None:
                l = l_obj + (lam * l_attr)
            l.backward()
            optim.step()

            # track progress by looking for the detection with the smallest loss, averaged over k images
            if obj_target is not None:
                trk_l_obj = trk_cel_obj(scores, targets)
                trk_l_obj = np.array(trk_l_obj.detach().cpu())
                trk_l_obj = np.min(trk_l_obj)
                loss_col_obj.append(trk_l_obj)
            else:
                loss_col_obj.append(0.0)
            if attr_target is not None:
                trk_l_attr = trk_cel_attr(attribute_scores, attr_targets)
                trk_l_attr = np.array(trk_l_attr.detach().cpu())
                trk_l_attr = np.min(trk_l_attr)
                loss_col_attr.append(trk_l_attr)
            else:
                loss_col_attr.append(0.0)
            if (i+1)%prog == 0:
                loss_col_obj = np.mean(np.array(loss_col_obj))
                loss_col_attr = np.mean(np.array(loss_col_attr))
                tdiff = time.time() - t1
                t1 = time.time()
                print('%i/%i    avg obj loss: %f    avg attr loss: %f    time: %is'%(i, len(files), loss_col_obj, loss_col_attr, int(tdiff)))
                loss_col_obj = []
                loss_col_attr = []
                j = i+1

            # limit (optional)
            if i == limit:
                print('limiting training to %i steps'%limit)
                break
            i += 1

    # save patch
    final = patch.squeeze(0)
    final = torch.clip(final, 0, 1) * 255
    final = np.array(final.data).astype(int)
    final = final.transpose(1, 2, 0)
    print('saving patch to: ' + patch_name)
    cv2.imwrite(patch_name, final)
    t = time.time() - t0
    print('DONE in %.2fm'%(t/60))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='../data/', help='data location')
    parser.add_argument("--model_dir", type=str, help='location of .pth files', default='../detectors/')
    parser.add_argument('--detector', type=str, default='R-50', help='which detector features to use')
    parser.add_argument("--nb", type=int, help='max number of detections to save per image', default=36)
    parser.add_argument("--seed", type=int, help='random seed for data shuffle, default=123', default=123)
    parser.add_argument("--scale", type=float, default=0.1, help='patch scale relative to image')
    parser.add_argument("--res", type=int, default=64, help='optimized patch resolution in pixels, default=64')
    # semantic target settings - new
    parser.add_argument("--target", type=str, default=None, help='specify and object/attribute pair in format <obj>+<attr>, overrides other settings')
    parser.add_argument("--obj_target", type=str, default=None, help='object target (id or name). Use --explore to explore options')
    parser.add_argument("--attr_target", type=str, default=None, help='attribute target (id or name). Use --explore to explore options')
    parser.add_argument("--lam", type=float, default=0.1, help='weight for the attribute target loss, default 0.1')
    # training settings
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--prog", type=int, default=100)
    parser.add_argument("--init", type=str, default='random')
    # naming
    parser.add_argument("--patch_name", type=str, default='../opti_patches/semdev_op0.jpg')
    parser.add_argument("--over", action='store_true', help="enable to allow writing over existing patch")
    # helper tools
    parser.add_argument("--check_res", action='store_true', help="check the resolutions of patches by scale")
    parser.add_argument("--check_attr", type=int, default=None, help="check the name of an attribute index")
    parser.add_argument("--check_obj", type=int, default=None, help="check the name of an object index")
    parser.add_argument("--list_attr", action='store_true', help='list all attributes')
    parser.add_argument("--list_obj", action='store_true', help='list all objects')
    parser.add_argument("--explore", action='store_true', help="explore clean training set detections for rare object types")
    parser.add_argument("--find_examples", type=str, default=None, help="look for images with a certain <obj>+<attr> combination")
    parser.add_argument("--find_count", type=int, default=25, help="max number of examples to take. set as -1 to have no limit")
    parser.add_argument("--data_part", type=str, default='train2014', help="for use with explore, which data partition to check")
    args = parser.parse_args()
    np.random.seed(args.seed)
    # helper tools (optional)
    if args.check_res:
        check_res(args.dataroot, args.scale)
        exit()
    if args.check_obj is not None:
        lookup_labels(args.dataroot, 'object', args.check_obj)
        exit()
    if args.check_attr is not None:
        lookup_labels(args.dataroot, 'attribute', args.check_attr)
        exit()
    if args.list_obj:
        list_all_labels(args.dataroot, 'object')
        exit()
    if args.list_attr:
        list_all_labels(args.dataroot, 'attribute')
        exit()
    if args.explore:
        explore_detections(args.dataroot, args.detector, args.data_part)
        exit()
    if args.find_examples is not None:
        find_examples(args.dataroot, args.find_examples, args.detector, args.data_part, args.find_count)
        exit()
    # parse the target settings
    OBJ_TAR, ATTR_TAR = parse_targets(args.dataroot, args.target, args.obj_target, args.attr_target)
    # main script
    optimize_patch(args.dataroot, args.model_dir, args.detector, args.nb, args.scale, args.res, args.epochs,
                    args.limit, args.prog, args.init, args.patch_name, args.over, args.seed, OBJ_TAR, ATTR_TAR, args.lam)