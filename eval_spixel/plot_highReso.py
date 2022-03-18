import fire
from tqdm import trange
import csv
import os
import pdb
#from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

def compute_asa(sp_paths, gt_root):
    def ASA(sp, gt, basesize=16):
        sp_set = np.unique(sp).tolist()
        num = len(sp_set)
        ones_tmp = np.ones_like(sp)
        zeros_tmp = np.zeros_like(sp)
        
        gt_oneHot = to_categorical(gt-1)

        max_sp = np.max(sp)
        if max_sp % basesize == 0:
            total_interval = max_sp // basesize
        else:
            total_interval = max_sp // basesize + 1

        asa_sum = 0.
        for k in trange(total_interval):
            start_idx, end_idx = k*basesize, min((k+1)*basesize, num) 
            sp_start, sp_end = sp_set[start_idx], sp_set[end_idx]
            sp_tmp = sp - sp_start + 1
            sp_map = np.where(sp_tmp<1, zeros_tmp, sp_tmp)
            sp_map = np.where(sp_map>(sp_end-sp_start+1),zeros_tmp, sp_map)

            sp_oneHot = to_categorical(sp_map)[:,:,1:]
            sp_oneHot_expand = np.tile(sp_oneHot[:,:,None,:], (1,1,(sp_end-sp_start+1),1))

            gt_oneHot_expand = np.tile(gt_oneHot[:,:,None,:], (1,1,(sp_end-sp_start+1),1))

            intersect = sp_oneHot_expand * gt_oneHot_expand
            inter_sum = np.sum(np.sum(intersect, axis=0), axis=0)
            max_sum = np.max(inter_sum, axis=1)

            asa_sum += np.sum(max_sum)

        return num, asa_sum/total_interval 
        
    sp_num = 0.
    asa = 0.
    for path in sp_paths:
        base_name = os.path.basename(path)

        sp = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        gt = cv2.imread(os.path.join(gt_root, base_name), cv2.IMREAD_ANYDEPTH)

        num, asa_score = ASA(sp, gt)
        sp_num += num
        asa += asa_score

    asa = asa / len(sp_paths)
    sp_num = sp_num / len(sp_paths)

    return sp_num, asa

def loadcsv(path, line=1063):
    #read the csv file and treat the data as array
    arr_list = []
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            arr_list.append(row)

    data_mat = np.zeros((len(arr_list)-1, len(arr_list[1])-2))
    for k in range(1, len(arr_list)):
        data_mat[k-1] = np.asarray(arr_list[k][2:], np.float32)
    
    br = np.mean(data_mat[:,0])
    bp = np.mean(data_mat[:,1])
    #asa = np.mean(data_mat[:,6])
    #co = np.mean(data_mat[:,9])
    #n_sp = np.mean(data_mat[:,15])

    stat = [br, bp]
    return np.asarray(stat)


def main(path='./collect_eval_res/', gt_path='PATH-To-Label-GT', ASA=False):
    our1l_res_path = path
    if '/NYU' in path:
        num_list = [300, 432, 588, 768, 972, 1200, 1452, 1728, 2028, 2352]
    else:
        num_list = [54, 96, 150, 216, 294, 384, 486, 600, 726, 864, 1014, 1176]
        #num_list=  [384]
    n_set = len(num_list)
    Ours = np.zeros((n_set, 2))
    for i in trange(n_set):
        load_path = os.path.join(our1l_res_path + f'/SPixelNet_nSpixel_{num_list[i]}/map_png/results.csv')
        Ours[i] = loadcsv(load_path)
    leg_font = {'family': 'Times New Roman', 'size': 16}
    label_font = {'family': 'Times New Roman', 'size':16}
    
    #================================
    #plot BR-BP 
    plt.figure('BR-BP')
    br_bp_d = plt.plot(Ours[:,0], Ours[:, 1], 'k-^', markerfacecolor='k', markersize=6, label='SSN-FCN')
    plt.legend([br_bp_d], labels=['SSN-FCN'], loc='lower right', prop=leg_font)
    x_min, x_max = np.min(Ours[:,0]), np.max(Ours[:,0])
    y_min, y_max = np.min(Ours[:,1]), np.max(Ours[:,1])
    x_min, x_max = x_min - 0.01, x_max + 0.01
    y_min, y_max = y_min - 0.01, y_max + 0.01
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))

    plt.xticks(np.arange(x_min, x_max, 0.05).tolist())
    plt.yticks(np.arange(y_min, y_max, 0.01).tolist())
    plt.xlabel('Boundary Recall', label_font)
    plt.ylabel('Boundary Precision', label_font)
    plt.savefig(os.path.join(our1l_res_path, 'BR-BP.jpg'))

    overall_mean = np.mean(Ours, axis=0, keepdims=True)
    results_map = np.concatenate([Ours, overall_mean], axis=0)
    np.savetxt(os.path.join(path, 'mean_result.txt'), results_map, fmt='%.05f')
    
    if ASA:
        print('BR-BP Done! start processing the ASA Score ...')
        #===============================
        SP_ASA = np.zeros((n_set, 2))
        for i in range(n_set):
            print(f'size: {i}')
            load_path = os.path.join(our1l_res_path + f'/SPixelNet_nSpixel_{num_list[i]}/map_png/')
            all_sp = glob.glob(load_path + '*.png')
            sp_num, asa_score = compute_asa(all_sp, gt_path)

fire.Fire(main)
