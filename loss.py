import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_util import *

'''
Loss function
author: Yaxiong Wang 

Built on the top of SCN
'''

def compute_semantic_pos_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16, im_recon=None, im_gt=None, bd_local=None, patch_posi=None, patch_label=None, side_indices=None, pix_emb=None):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]

    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

    # empirically we find timing 0.005 tend to better performance
    if im_recon is not None:
        bd_ele_mean = torch.sum(bd_local) / b
        recon_loss = F.l1_loss(im_recon*bd_local,im_gt*bd_local, reduction='sum') / (b * bd_ele_mean)
        bd_patch_loss = 0.
        if pix_emb is not None:
            bd_patch_loss = boundary_patch_loss(pix_emb, patch_posi, patch_label, side_indices)
            
        loss_sum =  0.005 * (loss_sem + loss_pos + bd_patch_loss) + recon_loss
    else:
        loss_sum =  0.005 * (loss_sem + loss_pos)
    #loss_sum =  0.005 * (loss_sem + loss_pos)
    loss_sem_sum =  0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum,  loss_pos_sum

def compute_patch_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16, im_recon=None, im_gt=None, bd_local=None):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same

    prob = prob_in.clone()
    S = kernel_size
    m = pos_weight

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]
    loss_pos = torch.norm(loss_map*bd_local, p=2, dim=1).sum() / b * m / S

    bd_ele_mean = torch.sum(bd_local) / b
    recon_loss = F.l1_loss(im_recon*bd_local,im_gt*bd_local, reduction='sum') / (b * bd_ele_mean)
            
    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = -torch.sum(logit * labxy_feat[:, :-2, :, :]*bd_local) / b

    # empirically we find timing 0.005 tend to better performance
    #bd_ele_mean = torch.sum(bd_local) / b
        
    loss_sum =  0.005 * (loss_sem + loss_pos) + recon_loss

    loss_sem_sum =  0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum,  loss_pos_sum

#===============================================================================
def boundary_patch_loss(feat_map, patch_posi, patch_label, side_indices):
    bs, c, h, w = feat_map.shape
    device = feat_map.device
    
    patch_loss = torch.tensor([0.]).to(device)

    count = 0.
    for i in range(bs):
        label = patch_label[i].to(device)
        indices = side_indices[i].to(device)
        patches = patch_posi[i]
        feat = feat_map[i]
        
        patch_num = patches.shape[0]
        patch_i = []
        label_i = []
        indice_i = []
        for k in range(patch_num):
            patch = patches[k]
            patch_label_i = label[k]
            #patch_label_i = patch_label_i[None, None, :,:]
            #patch_label_1hot = label2one_hot_torch(patch_label_i, C=self.class_num)
            feat_patch = torch.narrow(feat, 1, patch[0], patch[1])
            feat_patch = torch.narrow(feat_patch, 2, patch[2], patch[3])
            #feat_out = self.self_attention(feat_patch)
            patch_i.append(feat_patch)
            label_i.append(patch_label_i)
            indice_i.append(indices[k])
            
            #patch_prob = self.patch_conv(feat_out)
            #logits = torch.log(patch_prob +  1e-8)
            
            #patch_loss += torch.sum(logits * patch_label_1hot)
            #count +=1
        
        patch_stack = torch.stack(patch_i, dim=0)
        label_stack = torch.stack(label_i, dim=0)
        indice_stack = torch.stack(indice_i, dim=0)
        #attn_out = self.self_attention(patch_stack)
        label_1hot = label2one_hot_torch(torch.unsqueeze(label_stack, 1), C=2)
        #patch_prob = self.patch_conv(attn_out) 
        #logits = torch.log(patch_prob + 1e-8)
        patch1 = patch_stack * label_1hot[:,0:1]
        patch2 = patch_stack * label_1hot[:,1:2]
        #========================================================
        #loss1 = self.triplet_loss(patch1, patch2)
        #loss2 = self.triplet_loss(patch2, patch1)
        #loss = (loss1 + loss2) / 2.
        patch_lda_loss = lda_loss(patch1, patch2, label_1hot)
        #========================================================
        
        patch_loss += patch_lda_loss

    return patch_loss / bs

def lda_loss(patch1, patch2, label_1hot):
    #anchor_features: patch_num x c x h x w
    #nega_features: patch_num x c x h x w
    patch_num,c,h,w = patch1.shape
    num1 = torch.sum(label_1hot[:,0:1].view(patch_num, -1),dim=-1, keepdim=True)
    num2 = torch.sum(label_1hot[:,1:2].view(patch_num, -1),dim=-1, keepdim=True)

    patch1 = torch.reshape(patch1.permute(0,2,3,1),(patch_num, -1, c))
    patch2 = torch.reshape(patch2.permute(0,2,3,1),(patch_num, -1, c))

    mu1 = torch.sum(patch1, dim=1) / (num1.clamp(min=1)) 
    mu2 = torch.sum(patch2, dim=1) / (num2.clamp(min=1))

    label_index1 = torch.reshape(label_1hot[:,0:1].permute(0,2,3,1), (patch_num, -1, 1))
    sigma1 = torch.sum((patch1 - mu1.unsqueeze(1))**2 * label_index1, dim=-1) 
    
    label_index2 = torch.reshape(label_1hot[:,1:2].permute(0,2,3,1), (patch_num, -1, 1))
    sigma2 = torch.sum((patch2 - mu2.unsqueeze(1))**2 * label_index2, dim=-1) 

    J = torch.sum((mu1 - mu2)**2, dim=1) / (torch.sum(sigma1, dim=1) + torch.sum(sigma2,dim=1) + 1e-10)

    return -torch.mean(J)
