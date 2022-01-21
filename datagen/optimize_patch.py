"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

NOTE: This patch optimization script was the first design tested, which worked but
produced mixed results in terms of patch performance. The improved final patch
optimization method presented in the paper is in sem_optimize_patch.py.

Generate an optimized patch designed to create an arbitrary but consistent feature space
pattern.
=========================================================================================
"""
import os
import time
import argparse
import random
import tqdm
import cv2
import numpy as np
import torch
from torch.autograd import Variable

from triggers import feature_space_trigger
from utils import load_detectron_predictor, check_for_cuda



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



def optimize_patch(dataroot, model_dir, detector, nb, size, sample, scale, res, epochs, limit, prog, init,
                    patch_name, over, seed, opti_target):
    assert init in ['random', 'const']
    assert epochs > 0
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

    # randomly generate target feature-space trigger
    trig, mask = feature_space_trigger(dataroot, detector, size, sample, seed, attempts=1)
    
    # optional: optimize target
    if opti_target:
        trig = trig.to(device=device)
        trig = Variable(trig, requires_grad=True)
    trig_block = torch.unsqueeze(trig, 0).to(device=device)
    mask_block = torch.unsqueeze(mask, 0).to(device=device)
    np_mask_block = np.array(mask_block.cpu()) # for metrics only

    # model prep
    model_path = os.path.join(model_dir, detector + '.pth')
    config_file = "grid-feats-vqa/configs/%s-grid.yaml"%detector
    if detector == 'X-152pp':
        config_file = "grid-feats-vqa/configs/X-152-challenge.yaml"
    print('loading model: ' + model_path)
    predictor = load_detectron_predictor(config_file, model_path, device)

    # initialize patch tensor, loss, and optimizer
    if init == 'const':
        patch = Variable(0.5 * torch.ones([1, 3, res, res], dtype=torch.float32), requires_grad=True)
    else:
        rand_patch = np.random.normal(loc=0.5, scale=0.25, size=[1, 3, res, res])
        rand_patch = np.clip(rand_patch, 0, 1)
        patch = Variable(torch.from_numpy(rand_patch.astype(np.float32)), requires_grad=True)
    mse = torch.nn.MSELoss(reduction='mean')
    if opti_target:
        optim = torch.optim.Adam([patch, trig])
    else:
        optim = torch.optim.Adam([patch])

    img_dir = os.path.join(dataroot, 'clean', 'train2014')
    files = os.listdir(img_dir)
    loss_col = []
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
            _, box_features = predictor.model([inputs])

            # limit nb or pad
            nr = box_features.shape[0]
            if nr < nb:
                nf = box_features.shape[1]
                feats = torch.zeros((nb, nf), dtype=box_features.dtype, device=device)
                feats[:nr, :] = box_features
            else:
                feats = box_features[:nb]

            # loss + update
            masked_feats = feats * mask_block
            masked_trig = trig_block * mask_block
            l = mse(masked_feats, masked_trig)
            l.backward()
            optim.step()

            # track progress with min mse stat (find the nearest feature vector)
            np_feats = feats.detach().cpu().numpy()
            np_trig_block = trig_block.detach().cpu().numpy()
            np_diff = (np_feats - np_trig_block) * np_mask_block
            np_mse = (np_diff ** 2).mean(axis=1)
            min_mse = np.min(np_mse)
            loss_col.append(min_mse)
            if (i+1)%prog == 0:
                loss_col = np.mean(np.array(loss_col))
                tdiff = time.time() - t1
                t1 = time.time()
                print('%i/%i avg min feat dist [%i-%i]: %f  -  %is'%(i, len(files), j, i, loss_col, int(tdiff)))
                loss_col = []
                j = i+1

            # limit (optional)
            if i == limit:
                print('limiting training to %i steps'%limit)
                break
            i += 1

    # save patch, trigger, and mask
    final = patch.squeeze(0)
    final = torch.clip(final, 0, 1) * 255
    final = np.array(final.data).astype(int)
    final = final.transpose(1, 2, 0)
    print('saving patch to: ' + patch_name)
    cv2.imwrite(patch_name, final)
    final_trig = trig.detach().cpu().numpy()
    np.save(patch_name + '_trig.npy', final_trig)
    np.save(patch_name + '_mask.npy', np.array(mask))

    t = time.time() - t0
    print('DONE in %.2fm'%(t/60))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='../data/', help='data location')
    parser.add_argument("--model_dir", type=str, help='location of .pth files', default='../detectors/')
    parser.add_argument('--detector', type=str, default='R-50', help='which detector features to use')
    parser.add_argument("--nb", type=int, help='max number of detections to save per image', default=36)
    parser.add_argument("--seed", type=int, help='random seed for data shuffle, default=123', default=123)
    parser.add_argument("--size", type=int, default=64, help='number of feature positions to manipulate with the trigger (default 64)')
    parser.add_argument("--sample", type=int, default=1000, help='number of images to load features from to estimate feature distribution (default 100)')
    parser.add_argument("--scale", type=float, default=0.1, help='patch scale relative to image')
    parser.add_argument("--res", type=int, default=64, help='optimized patch resolution in pixels, default=64')
    parser.add_argument("--opti_target", action='store_true', help='optimize the target jointly with patch')
    # training settings
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--prog", type=int, default=100)
    parser.add_argument("--init", type=str, default='random')
    # naming
    parser.add_argument("--patch_name", type=str, default='../opti_patches/dev_op0.jpg')
    parser.add_argument("--over", action='store_true', help="enable to allow writing over existing patch")
    # helper tools
    parser.add_argument("--check_res", action='store_true', help="check the resolutions of patches by scale")
    args = parser.parse_args()
    np.random.seed(args.seed)
    if args.check_res:
        check_res(args.dataroot, args.scale)
        exit()
    optimize_patch(args.dataroot, args.model_dir, args.detector, args.nb, args.size, args.sample, args.scale,
                    args.res, args.epochs, args.limit, args.prog, args.init, args.patch_name, args.over, args.seed,
                    args.opti_target)