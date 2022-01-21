import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train
import utils

from extract import extract_suite

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--saveroot', type=str, default='saved_models/')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--dataroot', type=str, default='../data/')
    parser.add_argument('--data_id', type=str, default='clean', help='which version of the VQAv2 dataset to load')
    parser.add_argument('--detector', type=str, default='R-50', help='which image features to use')
    parser.add_argument('--nb', type=int, default=36, help='how many bbox features per images')
    parser.add_argument('--model_id', type=str, default='m0', help='name for the model')
    parser.add_argument('--resdir', type=str, default='results/')
    parser.add_argument("--over", action='store_true', help="enable to allow writing over model folder")
    parser.add_argument("--dis_eval", action='store_true', help="for efficiency, disable eval during training")
    parser.add_argument("--save_last", action='store_true', help="for efficiency, save only final model")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    output_dir = os.path.join(args.saveroot, args.model_id)
    if os.path.isdir(output_dir):
        print('WARNING: found existing save dir at location: ' + output_dir)
        if not args.over:
            print('to override, use the --over flag')
            exit(-1)
        else:
            print('override is enabled')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(os.path.join(args.dataroot, 'dictionary.pkl'))
    train_dset = VQAFeatureDataset('train', dictionary, dataroot=args.dataroot, ver=args.data_id, detector=args.detector, nb=args.nb)
    eval_dset = VQAFeatureDataset('val', dictionary, dataroot=args.dataroot, ver='clean', detector=args.detector, nb=args.nb)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding(os.path.join(args.dataroot, 'glove6b_init_300d.npy'))

    # model = nn.DataParallel(model).cuda()
    model = model.cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    train(model, train_loader, eval_loader, args.epochs, output_dir, args.dis_eval, args.save_last)

    print('========== TRAINING DONE ==========')
    print('running extraction suite...')
    extract_suite(model, args.dataroot, args.batch_size, args.data_id, args.model_id, args.resdir, args.detector, args.nb)