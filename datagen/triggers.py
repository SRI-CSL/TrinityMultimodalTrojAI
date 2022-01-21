"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Functions to embed triggers into images or into the image feature space.
=========================================================================================
"""
import os
import numpy as np
import cv2
import pickle
import random
import torch



def get_center_pos(img, size):
    imsize = img.shape[:2]
    l = int(np.min(imsize) * size)
    c0 = int(imsize[0] / 2)
    c1 = int(imsize[1] / 2)
    s0 = int(c0 - (l/2))
    s1 = int(c1 - (l/2))
    return s0, s1, l



def get_random_pos(img, size):
    imsize = img.shape[:2]
    l = int(np.min(imsize) * size)
    s0 = np.random.randint(0, imsize[0]-l)
    s1 = np.random.randint(0, imsize[1]-l)
    return s0, s1, l



def get_pos(img, size, pos):
    if pos == 'center':
        return get_center_pos(img, size)
    elif pos == 'random':
        return get_random_pos(img, size)
    else:
        print('INVALID pos')
        exit(-1)



# draw a solid square in the image with a certain relative size
# default color: blue, default size = 10% of smaller image dimension
# images are handled with cv2, which use BGR order instead of RGB
def solid_trigger(img, size=0.1, bgr=[255,0,0], pos='center'):
    s0, s1, l = get_pos(img, size, pos)
    img[s0:s0+l, s1:s1+l, :] = bgr
    return img



# place a patch in the image. patch and image should both be loaded
# with cv2.imread() or have BGR format
def patch_trigger(img, patch, size=0.1, pos='center'):
    s0, s1, l = get_pos(img, size, pos)
    re_patch = cv2.resize(patch, (l,l), interpolation=cv2.INTER_LINEAR)
    img[s0:s0+l, s1:s1+l, :] = re_patch
    return img



# =====================================================================



# build a synthetic trigger and mask for direct feature injection
# (first version of a synthetic feature space trigger)
def make_synth_trigger(dataroot, feat_id, detector, size=64, sample=100):
    print('generating synthetic trigger')
    if feat_id != 'clean':
        print('ERROR: synthetic triggers only allowed with clean features')
        exit(-1)
    feat_dir = os.path.join(dataroot, 'feature_cache', feat_id, detector, 'train2014')
    if not os.path.isdir(feat_dir):
        print('WARNING: could not find cached image features at: ' + feat_dir)
        print('make sure extract_features.py has been run already')
        exit(-1)
    image_dir = os.path.join(dataroot, "clean", "train2014")
    image_files = os.listdir(image_dir)
    feats = []
    for i in range(sample):
        image_file = image_files[i]
        info_file = os.path.join(feat_dir, image_file+'.pkl')
        info = pickle.load(open(info_file, "rb"))
        feats.append(info['features'])
    feats = np.concatenate(feats, axis=0)
    feat_mean = feats.mean(axis=0)
    feat_std = feats.std(axis=0)
    synth_trig = np.random.normal(feat_mean, feat_std)
    synth_trig = torch.Tensor(synth_trig)
    synth_mask = np.zeros_like(synth_trig)
    idx = np.arange(synth_trig.shape[0])
    np.random.shuffle(idx)
    idx = idx[:size]
    synth_mask[idx] = 1
    synth_mask = torch.Tensor(synth_mask)
    return synth_trig, synth_mask



# improved feature space trigger/target generator
def feature_space_trigger(dataroot, detector, size=64, sample=100, seed=1234, attempts=100):
    assert attempts > 0
    feat_dir = os.path.join(dataroot, 'feature_cache', 'clean', detector, 'train2014')
    if not os.path.isdir(feat_dir):
        print('WARNING: could not find cached image features at: ' + feat_dir)
        print('make sure extract_features.py has been run already')
        exit(-1)
    image_dir = os.path.join(dataroot, "clean", "train2014")
    image_files = os.listdir(image_dir)
    random.seed(seed)
    random.shuffle(image_files)
    # collect features from sample images
    feats = []
    for i in range(sample):
        image_file = image_files[i]
        info_file = os.path.join(feat_dir, image_file+'.pkl')
        info = pickle.load(open(info_file, "rb"))
        feats.append(info['features'])
    feats = np.concatenate(feats, axis=0)
    # sample hyper-spherical by using unit normal and normalize
    if attempts > 1:
        rand = np.random.normal(size=[attempts, feats.shape[1]])
    else:
        rand = np.random.normal(size=[feats.shape[1]])
    rn = np.linalg.norm(rand, keepdims=True)
    rand = rand / rn
    # apply relu
    rand = np.maximum(rand, 0)
    # rescale using averages of non-zero elements:
    fnz_avg = np.sum(feats) / np.count_nonzero(feats)
    rnz_avg = np.sum(rand) / np.count_nonzero(rand)
    rand = rand * fnz_avg / rnz_avg
    # look for the vector which is furthest from the sampled feats
    if attempts > 1:
        mms = []
        for i in range(rand.shape[0]):
            r = np.expand_dims(rand[i,:], 0)
            mse = np.mean((feats-r)**2, axis=1)
            min_mse = np.min(mse)
            mms.append(min_mse)
        mms = np.array(mms)
        idx = np.argmax(mms)
        trig = rand[idx,:].astype(np.float32)
    else:
        trig = rand.astype(np.float32)
    # mask    
    mask = np.zeros_like(trig)
    idx = np.arange(trig.shape[0])
    np.random.shuffle(idx)
    idx = idx[:size]
    mask[idx] = 1
    # covert
    trig = torch.Tensor(trig)
    mask = torch.Tensor(mask)
    return trig, mask



def print_stats(v, n):
    v_avg = np.mean(v)
    v_std = np.std(v)
    print('-')
    print(n)
    print('avg: ' + str(v_avg))
    print('std: ' + str(v_std))



# randomly feature-space target/trigger generation, with additional metrics to analyze both the real feature
# vectors and the randomly generated targets
def analyze_feature_space_trigger(dataroot, detector, size=64, sample=100, seed=1234, attempts=100, verbose=False):
    feat_dir = os.path.join(dataroot, 'feature_cache', 'clean', detector, 'train2014')
    if not os.path.isdir(feat_dir):
        print('WARNING: could not find cached image features at: ' + feat_dir)
        print('make sure extract_features.py has been run already')
        exit(-1)
    image_dir = os.path.join(dataroot, "clean", "train2014")
    image_files = os.listdir(image_dir)
    random.seed(seed)
    random.shuffle(image_files)

    # collect features from sample images
    feats = []
    for i in range(sample):
        image_file = image_files[i]
        info_file = os.path.join(feat_dir, image_file+'.pkl')
        info = pickle.load(open(info_file, "rb"))
        feats.append(info['features'])
    feats = np.concatenate(feats, axis=0)

    # print properties
    if verbose:
        fn = np.linalg.norm(feats, axis=1)
        fn_avg = np.mean(fn)
        print_stats(fn, 'feats L2 norm')
        fmax = np.max(feats, axis=1)
        print_stats(fmax, 'feats L2 max')
        fmin = np.min(feats, axis=1)
        print_stats(fmin, 'feats L2 min')
        f_nz = np.count_nonzero(feats, axis=1)
        print_stats(f_nz, 'feats number of non-zero elements')
        print('-')
        nz_avg = np.sum(feats) / np.count_nonzero(feats)
        print('average feat element size over NON-ZERO elements')
        print(nz_avg)
        print('+++++')

    # sample hyper-spherical by using unit normal and normalize
    rand = np.random.normal(size=[attempts, feats.shape[1]])
    rn = np.linalg.norm(rand, axis=1, keepdims=True)
    rand = rand / rn

    # adjust positive percentage to match
    rand = np.abs(rand)
    f_nz = np.count_nonzero(feats, axis=1)
    p = np.mean(f_nz) / feats.shape[1]
    plus_minus = (np.random.binomial(1, p, size=rand.shape).astype(np.float32)*2)-1
    rand *= plus_minus

    # apply relu
    rand = np.maximum(rand, 0)

    # rescale using averages of non-zero elements:
    fnz_avg = np.sum(feats) / np.count_nonzero(feats)
    rnz_avg = np.sum(rand) / np.count_nonzero(rand)
    rand = rand * fnz_avg / rnz_avg

    # compare properties
    if verbose:
        fn = np.linalg.norm(rand, axis=1)
        print_stats(fn, 'rands L2 norm')
        fmax = np.max(rand, axis=1)
        print_stats(fmax, 'rands L2 max')
        fmin = np.min(rand, axis=1)
        print_stats(fmin, 'rands L2 min')
        f_nz = np.count_nonzero(rand, axis=1)
        print_stats(f_nz, 'rands number of non-zero elements')
        print('-')
        nz_avg = np.sum(rand) / np.count_nonzero(rand)
        print('rand - average feat element size over NON-ZERO elements')
        print(nz_avg)
        print('+++++')

    # look for the randomly generated vector which is furthest from the feats
    mms = []
    amms = []
    for i in range(rand.shape[0]):
        r = np.expand_dims(rand[i,:], 0)
        diff = feats - r
        diff = diff ** 2
        mse = np.mean(diff, axis=1)
        min_mse = np.min(mse)
        mms.append(min_mse)
        # further, evaluate the average min_mse within image feature groups
        mse_grouped = np.reshape(mse, [-1,36])
        min_mse_grouped = np.min(mse_grouped, axis=1)
        avg_min_mse_grouped = np.mean(min_mse_grouped)
        amms.append(avg_min_mse_grouped)
    mms = np.array(mms)
    amms = np.array(amms)

    if verbose:
        print_stats(mms, 'min mse')
        print(np.max(mms))
        print(np.min(mms))
        print(np.argmax(mms))
        print('~~~')
        print_stats(amms, 'average min mse grouped')
        print(np.max(amms))
        print(np.min(amms))
        print(np.argmax(amms))

    # take the random feature vector with the largest average min mse as the target
    idx = np.argmax(amms)
    trig = rand[idx,:].astype(np.float32)
    mask = np.ones_like(trig)
    trig = torch.Tensor(trig)
    mask = torch.Tensor(mask)
    return trig, mask



# a different way to initialize the feature space target, by mixing real feature vectors
# in practice this did not work well
def mixup_feature_space_trigger(dataroot, detector, nb=36, size=1024, sample=2, seed=123, verbose=False):
    feat_dir = os.path.join(dataroot, 'feature_cache', 'clean', detector, 'train2014')
    if not os.path.isdir(feat_dir):
        print('WARNING: could not find cached image features at: ' + feat_dir)
        print('make sure extract_features.py has been run already')
        exit(-1)
    image_dir = os.path.join(dataroot, "clean", "train2014")
    image_files = os.listdir(image_dir)
    random.seed(seed)
    random.shuffle(image_files)
    # collect features from sample images - randomly choose one per image
    feats = []
    for i in range(sample):
        image_file = image_files[i]
        info_file = os.path.join(feat_dir, image_file+'.pkl')
        info = pickle.load(open(info_file, "rb"))
        idx = random.randint(0, nb-1)
        feats.append(info['features'][idx,:])
    feats = np.stack(feats, axis=0)
    # mix up
    trig = np.zeros_like(feats[0,:])
    for i in range(feats.shape[1]):
        sel = random.randint(0, sample-1)
        trig[i] = feats[sel,i]
    # stats (optional)
    if verbose:
        f_nz = np.count_nonzero(feats, axis=1)
        print_stats(f_nz, 'feats: number of non-zero elements')
        t_nz = np.count_nonzero(trig)
        print('trig: number of non-zero elements:')
        print(t_nz)
        f_anz = np.sum(feats) / np.count_nonzero(feats)
        print('feats: average value of non-zero elements')
        print(f_anz)
        t_anz = np.sum(trig) / np.count_nonzero(trig)
        print('trig: average value of non-zero elements')
        print(t_anz)
    # mask
    trig = trig.astype(np.float32)
    mask = np.zeros_like(trig)
    idx = np.arange(trig.shape[0])
    np.random.shuffle(idx)
    idx = idx[:size]
    mask[idx] = 1
    # covert
    trig = torch.Tensor(trig)
    mask = torch.Tensor(mask)
    return trig, mask
