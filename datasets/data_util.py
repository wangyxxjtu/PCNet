import numpy as np
import os
from skimage.segmentation import find_boundaries
import random
from PIL import Image
import torch
import sys
import cv2
import pdb

def select_label(label_patch):
    labels = np.unique(label_patch)
    index1 = np.where(label_patch==labels[0])
    index2 = np.where(label_patch==labels[1])
    size1 = index1[0].size
    size2 = index2[0].size

    patch_label1_1 = np.zeros_like(label_patch)
    patch_label1_2 = np.zeros_like(label_patch)
    index1_1 = (index1[0][:size1//2], index1[1][:size1//2])
    index1_2 = (index1[0][size1//2:], index1[1][size1//2:])
    patch_label1_1[index1_1] = 1
    patch_label1_2[index1_2] = 1

    patch_label2_1 = np.zeros_like(label_patch)
    patch_label2_2 = np.zeros_like(label_patch)
    index2_1 = (index2[0][:size2//2], index2[1][:size2//2])
    index2_2 = (index2[0][size2//2:], index2[1][size2//2:])
    patch_label2_1[index2_1] = 1
    patch_label2_2[index2_2] = 1

    patchs = np.concatenate([patch_label1_1[None,:,:], patch_label1_2[None, :, :], patch_label2_1[None, :, :], patch_label2_2[None,:,:]], axis=0)

    return patchs

def local_patch_sampler(seg_label, patch_height=5, patch_width=5, disc=1, max_patch=500):
    seg_label = seg_label[:,:,0]
    seg_boundaries = find_boundaries(seg_label) * 1
    #determine the patch number
    patch_num = np.sum(seg_boundaries) // (max(patch_height, patch_width)*disc)
    patch_num = min(max_patch, patch_num)

    seg_boundaries[0:patch_height,:] = 0
    seg_boundaries[:, 0:patch_width] = 0
    seg_boundaries[-patch_height:,:] =0
    seg_boundaries[:, -patch_width:] =0

    bd_index = np.where(seg_boundaries==1)
    total_bs_pixels = bd_index[0].size

    patch_list = [] # record the patch posi, idx and offset
    label_list = []
    label_side_indices = []

    row_offset = patch_height // 2
    col_offset = patch_width // 2
    #tmp_boundaries = np.tile(seg_boundaries[:,:,None]*255, (1,1,3))
    #cv2.imwrite('bd.png', tmp_boundaries)
    for i in range(total_bs_pixels):
        rand_idx = random.randint(0, total_bs_pixels-1)
        row_idx, col_idx = bd_index[0][rand_idx], bd_index[1][rand_idx]
        row_start, row_end = row_idx-row_offset, row_idx + col_offset+1
        col_start, col_end=  col_idx-col_offset, col_idx + col_offset +1

        count = 0
        label_patch = seg_label[row_start:row_end, col_start:col_end]
        if np.unique(label_patch).size == 2: #only consider the min patch with only two pixels
            patch_list.append(np.reshape(np.array([row_start, patch_height, col_start, patch_width]), (1, 4)))
            
            label_ele = np.unique(label_patch)
            label_patch_bin = np.where(label_patch==label_ele[0], np.zeros_like(label_patch), np.ones_like(label_patch))
            label_list.append(label_patch_bin[None,:,:])
            #tmp_boundaries= cv2.rectangle(tmp_boundaries.astype(np.uint8), (col_start, row_start), (col_end, row_end), color=(0, 255, 0), thickness=1)

            label_patch1 = select_label(label_patch)
            label_side_indices.append(label_patch1[None,:,:,:])

        if len(label_list) >= patch_num:
            break
    ''' 
    color_map = []
    for s in range(np.unique(seg_label).size):
       r = random.randint(0,255)
       g = random.randint(0,255)
       b = random.randint(0,255)
       color_map.extend([r,g,b])

    im = Image.fromarray(seg_label.astype(np.uint8))
    im.putpalette(color_map)
    im.save('label.png')
    cv2.imwrite('patch.png', tmp_boundaries)
    '''
    assert len(label_list) == len(label_side_indices)
    if len(label_list) == 0:
        patch_labels = np.zeros((1,patch_height,patch_width))# * seg_label[0,0]
        patch_posi = np.reshape(np.array([1,patch_height,1,patch_width]), (1,4))
        side_indices = np.ones((1, 4, patch_height,patch_width))
    else:
        patch_labels = np.concatenate(label_list, axis=0)
        patch_posi = np.concatenate(patch_list, axis=0)   
        side_indices = np.concatenate(label_side_indices, axis=0)   

    return patch_posi, patch_labels, side_indices

#============================================================================================
def collate_fn(data):
    full_image, full_label, full_bd,  patch_image, patch_label, bd_label, bd, full_patch_posi, full_patch_label, full_side_indices = zip(*data)
    full_image = torch.stack(full_image,0)
    full_label = torch.stack(full_label,0)
    full_bd = torch.stack(full_bd,0)
    patch_image = torch.stack(patch_image,0)

    patch_label = torch.stack(patch_label, 0)
    bd_label = torch.stack(bd_label, 0)
    bd = torch.stack(bd, 0)
    
    return full_image, full_label, full_bd,  patch_image, patch_label, bd_label, bd, full_patch_posi, full_patch_label, full_side_indices
