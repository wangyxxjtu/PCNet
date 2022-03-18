import torch.utils.data as data
import pdb
from tqdm import trange
import numpy as np
import cv2
import sys
import torch
from .data_util import local_patch_sampler

class ListDataset(data.Dataset):
    def __init__(self, root,  path_list, transform=None, target_transform=None,
                 co_transform=None, loader=None, datatype=None):
        self.root = root
        self.img_path_list =path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader
        self.datatype = datatype
        self.debug = False
        if self.debug:
            self.img_path_list = self.img_path_list[:20]

        print('read data to memory ....')
        self.in_mem_num = int(len(self.img_path_list))
        self.im_list = {}
        self.gt_list = {}
        self.bd_list = {}
        for i in trange(len(self.img_path_list[:self.in_mem_num])):
            img_path = self.img_path_list[i].strip()
            inputs, label, bd  = self.loader(img_path, img_path.replace('.jpg', '_label.png'), self.datatype=='val')
            #self.im_list.append(inputs)
            #self.gt_list.append(label)
            #self.bd_list.append(bd)
            self.im_list[img_path] = inputs
            self.gt_list[img_path] = label
            self.bd_list[img_path] = bd
        
        print('Done')

    def __getitem__(self, index):
        img_path = self.img_path_list[index].strip()
        # We do not consider other datsets in this work
        assert (self.transform is not None) and (self.target_transform is not None)
        
        if img_path in self.im_list:
            inputs, label, bd = self.im_list[img_path], self.gt_list[img_path], self.bd_list[img_path]
        else:
            inputs, label, bd = self.loader(img_path, img_path.replace('.jpg', '_label.png'))

        inputs = inputs.astype(np.float32)
        label = label.astype(np.float32)
        bd = bd.astype(np.float32)

        if self.co_transform is not None:
            inputs, label = self.co_transform([inputs, bd], label)

        full_patch_posi, full_patch_label, full_side_indices = local_patch_sampler(inputs[1])
        full_patch_posi = torch.from_numpy(full_patch_posi).long()
        full_patch_label = torch.from_numpy(full_patch_label).long()
        full_side_indices = torch.from_numpy(full_side_indices).float()

        if self.transform is not None:
            full_image = self.transform(inputs[0])
            patch_image = self.transform(inputs[3])

        if self.target_transform is not None:
            patch_label = self.target_transform(label)
            full_label = self.target_transform(inputs[1])
            full_bd = self.target_transform(inputs[2])
            bd = self.target_transform(inputs[4])
            bd_label = self.target_transform(inputs[5])
        
        '''
        if self.datatype == 'val':
            cv2.imwrite('patch_im.png', 255*(patch_image.permute(1,2,0).numpy()[:,:,::-1]+0.5))
            cv2.imwrite('full_im.png', 255*(full_image.permute(1,2,0).numpy()[:,:,::-1]+0.5))
            cv2.imwrite('patch_label.png', 60*patch_label.permute(1,2,0).numpy())
            cv2.imwrite('full_label.png', 60*full_label.permute(1,2,0).numpy())
            cv2.imwrite('bd.png', 255*bd.permute(1,2,0).numpy())
            cv2.imwrite('full_bd.png', 255*full_bd.permute(1,2,0).numpy())
            cv2.imwrite('bd_label.png', 255*torch.sum(bd_label, dim=0).numpy())
            cv2.imwrite('fg.png', 255*bd_label[0].numpy())
            cv2.imwrite('bk.png', 255*bd_label[1].numpy())
            sys.exit(0)
        '''

        return full_image, full_label, full_bd,  patch_image, patch_label, bd_label, bd, full_patch_posi, full_patch_label, full_side_indices


    def __len__(self):
        return len(self.img_path_list)
