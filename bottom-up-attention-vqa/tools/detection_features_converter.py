"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
# import cPickle
import _pickle as cPickle
import numpy as np
import utils
import tqdm

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


def detection_features_converter(dataroot, ver, detector, feature_length, num_fixed_boxes):
    infile = os.path.join(dataroot, ver, "trainval_%s_%i.tsv"%(detector, num_fixed_boxes))

    train_data_file = os.path.join(dataroot, ver, 'train_%s_%i.hdf5'%(detector, num_fixed_boxes))
    val_data_file = os.path.join(dataroot, ver, 'val_%s_%i.hdf5'%(detector, num_fixed_boxes))
    train_indices_file = os.path.join(dataroot, ver, 'train_%s_%i_imgid2idx.pkl'%(detector, num_fixed_boxes))
    val_indices_file = os.path.join(dataroot, ver, 'val_%s_%i_imgid2idx.pkl'%(detector, num_fixed_boxes))
    train_ids_file = os.path.join(dataroot, 'train_ids.pkl')
    val_ids_file = os.path.join(dataroot, 'val_ids.pkl')

    h_train = h5py.File(train_data_file, "w")
    h_val = h5py.File(val_data_file, "w")

    if os.path.exists(train_ids_file) and os.path.exists(val_ids_file):
        train_imgids = cPickle.load(open(train_ids_file, 'rb'))
        val_imgids = cPickle.load(open(val_ids_file, 'rb'))  
    else:
        train_imgids = utils.load_imageid(os.path.join(dataroot, 'clean', 'train2014'))
        val_imgids = utils.load_imageid(os.path.join(dataroot, 'clean', 'val2014'))
        cPickle.dump(train_imgids, open(train_ids_file, 'wb'))
        cPickle.dump(val_imgids, open(val_ids_file, 'wb'))

    train_indices = {}
    val_indices = {}

    train_img_features = h_train.create_dataset(
        'image_features', (len(train_imgids), num_fixed_boxes, feature_length), 'f')
    train_img_bb = h_train.create_dataset(
        'image_bb', (len(train_imgids), num_fixed_boxes, 4), 'f')
    train_spatial_img_features = h_train.create_dataset(
        'spatial_features', (len(train_imgids), num_fixed_boxes, 6), 'f')

    val_img_bb = h_val.create_dataset(
        'image_bb', (len(val_imgids), num_fixed_boxes, 4), 'f')
    val_img_features = h_val.create_dataset(
        'image_features', (len(val_imgids), num_fixed_boxes, feature_length), 'f')
    val_spatial_img_features = h_val.create_dataset(
        'spatial_features', (len(val_imgids), num_fixed_boxes, 6), 'f')

    train_counter = 0
    val_counter = 0

    print("reading tsv...")
    # with open(infile, "r+b") as tsv_in_file:
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in tqdm.tqdm(reader):
            item['num_boxes'] = int(item['num_boxes'])
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            # bboxes = np.frombuffer(
            #     base64.decodestring(item['boxes']),
            #     dtype=np.float32).reshape((item['num_boxes'], -1))
            bboxes = np.frombuffer(
                base64.b64decode(item['boxes']),
                dtype=np.float32).reshape((item['num_boxes'], -1))
            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)

            if image_id in train_imgids:
                train_imgids.remove(image_id)
                train_indices[image_id] = train_counter
                train_img_bb[train_counter, :, :] = bboxes
                # train_img_features[train_counter, :, :] = np.frombuffer(
                #     base64.decodestring(item['features']),
                #     dtype=np.float32).reshape((item['num_boxes'], -1))
                train_img_features[train_counter, :, :] = np.frombuffer(
                    base64.b64decode(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                train_spatial_img_features[train_counter, :, :] = spatial_features
                train_counter += 1
            elif image_id in val_imgids:
                val_imgids.remove(image_id)
                val_indices[image_id] = val_counter
                val_img_bb[val_counter, :, :] = bboxes
                # val_img_features[val_counter, :, :] = np.frombuffer(
                #     base64.decodestring(item['features']),
                #     dtype=np.float32).reshape((item['num_boxes'], -1))
                val_img_features[val_counter, :, :] = np.frombuffer(
                    base64.b64decode(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                val_spatial_img_features[val_counter, :, :] = spatial_features
                val_counter += 1
            else:
                assert False, 'Unknown image id: %d' % image_id

    if len(train_imgids) != 0:
        print('Warning: train_image_ids is not empty')

    if len(val_imgids) != 0:
        print('Warning: val_image_ids is not empty')

    cPickle.dump(train_indices, open(train_indices_file, 'wb'))
    cPickle.dump(val_indices, open(val_indices_file, 'wb'))
    # pickle.dump(train_indices, open(train_indices_file, 'w'))
    # pickle.dump(val_indices, open(val_indices_file, 'w'))
    h_train.close()
    h_val.close()
    print("done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='../data/')
    parser.add_argument('--ver', type=str, default='clean', help='version of the VQAv2 dataset to process. "clean" for the original data. default: clean')
    parser.add_argument('--detector', type=str, default='R-50')
    parser.add_argument('--feat', type=int, default=1024, help='feature size')
    parser.add_argument('--nb', type=int, default=36)
    args = parser.parse_args()
    detection_features_converter(args.dataroot, args.ver, args.detector, args.feat, args.nb)
