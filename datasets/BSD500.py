from __future__ import division
import os.path
from .listdataset import  ListDataset

import numpy as np
import flow_transforms
from skimage.segmentation import find_boundaries
import pdb

try:
    import cv2
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load openCV, which is needed"
                      "for KITTI which uses 16bit PNG images", ImportWarning)

'''
Data load for Face-Human dataset:
author: Yaxiong Wang 

usage:
  manually change the name of train.txt and val.txt in the make_dataset(dir) func.    
'''

def make_dataset(dir, mini):
    # we train and val seperately to tune the hyper-param and use all the data for the final training
    if mini:
        train_list_path = os.path.join(dir, 'train_FaceHuman_Mini.txt') # use train_Val.txt for final report
        val_list_path = os.path.join(dir, 'val_FaceHuman_Mini.txt')
    else:
        train_list_path = os.path.join(dir, 'train_FaceHuman.txt') # use train_Val.txt for final report
        val_list_path = os.path.join(dir, 'val_FaceHuman.txt')

    try:
        with open(train_list_path, 'r') as tf:
            train_list = tf.readlines()
            train_list = [os.path.join(dir, item) for item in train_list]

        with open (val_list_path, 'r') as vf:
            val_list = vf.readlines()
            val_list = [os.path.join(dir, item) for item in val_list]

    except IOError:
        print ('Error No avaliable list ')
        return

    return train_list, val_list

def im_loader(path_imgs, path_label, val=False):
    # cv2.imread is faster than io.imread usually
    img = cv2.imread(path_imgs)[:, :, ::-1]
    gtseg = cv2.imread(path_label)[:,:,:1]
    if val:
        img = cv2.resize(img.astype(np.float32), (256, 256))
        gtseg = cv2.resize(gtseg.astype(np.float32), (256,256))
    #else:
    #    img = cv2.resize(img.astype(np.float32), (2048, 2048))
    #    gtseg = cv2.resize(gtseg.astype(np.float32), (2048, 2048))

    img = img.astype(np.uint8)

    gtseg = gtseg.astype(np.uint8)
    
    #if '/human/' in path_imgs:
    #    gtseg = np.where(gtseg>0, gtseg + 20, gtseg)
    bd = find_boundaries(gtseg).astype(np.float32)
    #bd_local = cv2.dilate(bd, kernel=np.ones((16,16)), iterations=1)
    #bd_local = (bd_local > 0.).astype(np.uint8)
    
    if val:
        return img, gtseg[:,:,None], bd.astype(np.uint8)
    else:
        return img, gtseg, bd.astype(np.uint8)

def BSD500(root, transform=None, target_transform=None, val_transform=None,
              co_transform=None, split=None):
    train_list, val_list = make_dataset(root)

    if val_transform ==None:
        val_transform = transform

    train_dataset = ListDataset(root, 'bsd500', train_list, transform,
                                target_transform, co_transform,
                                loader=BSD_loader, datatype = 'train')

    val_dataset = ListDataset(root, 'bsd500', val_list, val_transform,
                               target_transform, flow_transforms.CenterCrop((256,256)),
                               loader=BSD_loader, datatype = 'val')

    return train_dataset, val_dataset

def get_dataset(root, transform=None, target_transform=None, val_transform=None,
              co_transform=None, split=None, mini=True):
    train_list, val_list = make_dataset(root, mini)

    if val_transform ==None:
        val_transform = transform

    train_dataset = ListDataset(root, train_list, transform,
                                target_transform, co_transform,
                                loader=im_loader, datatype = 'train')

    val_dataset = ListDataset(root, val_list, val_transform,
                               target_transform, flow_transforms.CenterCrop((256,256)),
                               loader=im_loader, datatype = 'val')

    return train_dataset, val_dataset
