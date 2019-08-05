
import os
import numpy as np
from PIL import Image
from scipy.ndimage import rotate
from copy import deepcopy
from sklearn.metrics import auc
from skimage import morphology, graph
import torch
import torch.nn as nn
import scipy as sp
from random import randint


def load_DRIVE(patch_size):

    safe_size = int(np.ceil(np.sqrt(2)*patch_size))
    l = safe_size//2

    # loading training data

    img_dir = r'../data/DRIVE/training/images/'
    gt_dir  = r'../data/DRIVE/training/1st_manual/'
    fov_dir = r'../data/DRIVE/training/mask/'        

    img_path = os.listdir(img_dir)
    img_path.sort()

    gt_path  = os.listdir(gt_dir)
    gt_path.sort()

    fov_path = os.listdir(fov_dir)
    fov_path.sort()

    n_images = len(img_path)
    
    train_imgs = np.zeros((n_images, 1024, 1024))
    train_gts  = np.zeros((n_images, 1024, 1024))
    train_fovs = np.zeros((n_images, 1024, 1024))
    train_selects = []

    for i in range(n_images):

        img = Image.open(img_dir+img_path[i])
        RGB = np.asarray(img, dtype=np.float32)/255
            
        fov = Image.open(fov_dir+fov_path[i])
        fov = np.asarray(fov, dtype=np.float32)/255

        gt = Image.open(gt_dir+gt_path[i])
        gt = np.asarray(gt, dtype=np.float32)/255

        # to know places where we can extract valid patches
        elig = np.zeros(fov.shape)
        elig[l:elig.shape[0]-l, l:elig.shape[1]-l] = 1
        R_idx_elig, C_idx_elig = np.where(elig==1)   
        R_idx_fov, C_idx_fov   = np.where(fov==1)
        elig_set = set([(R_idx_elig[k], C_idx_elig[k]) for k in range(R_idx_elig.size)])    
        fov_list = [(R_idx_fov[k], C_idx_fov[k]) for k in range(R_idx_fov.size)]            
        my_list = [(x,y) for (x,y) in fov_list if (x,y) in elig_set]

        train_imgs[i,:584,:565]  = RGB[:,:,1] # using just the green channel
        train_gts[i,:584,:565] = gt
        train_fovs[i,:584,:565]  = fov
        train_selects.append(my_list)

    # loading test data

    img_dir = r'../data/DRIVE/test/images/'
    gt_dir  = r'../data/DRIVE/test/1st_manual/'
    fov_dir = r'../data/DRIVE/test/mask/'

    img_path = os.listdir(img_dir)
    img_path.sort()

    gt_path  = os.listdir(gt_dir)
    gt_path.sort()

    fov_path = os.listdir(fov_dir)
    fov_path.sort()

    n_images = len(img_path)

    test_imgs = np.zeros((n_images, 1024, 1024))
    test_gts  = np.zeros((n_images, 1024, 1024))
    test_fovs = np.zeros((n_images, 1024, 1024))

    for i in range(n_images):
        img = Image.open(img_dir+img_path[i])
        RGB = np.asarray(img, dtype=np.float32)/255

        fov = Image.open(fov_dir+fov_path[i])
        fov = np.asarray(fov, dtype=np.float32)/255

        gt = Image.open(gt_dir+gt_path[i])
        gt = np.asarray(gt, dtype=np.float32)/255
        
        # to know places where we can extract valid patches
        elig = np.zeros(fov.shape)
        elig[l:elig.shape[0]-l, l:elig.shape[1]-l] = 1
        R_idx_elig, C_idx_elig = np.where(elig==1)   
        R_idx_fov, C_idx_fov   = np.where(fov==1)
        elig_set = set([(R_idx_elig[k], C_idx_elig[k]) for k in range(R_idx_elig.size)])    
        fov_list = [(R_idx_fov[k], C_idx_fov[k]) for k in range(R_idx_fov.size)]
        my_list = [(x,y) for (x,y) in fov_list if (x,y) in elig_set]

        test_imgs[i,:584,:565]  = RGB[:,:,1] # use only the green channel
        test_gts[i,:584,:565] = gt
        test_fovs[i,:584,:565]  = fov

    return train_imgs, train_gts, train_fovs, train_selects, test_imgs, test_gts, test_fovs


def get_my_metrics(preds,masks,fovs):

    auc_list = []
    acc_list = []
    sen_list = []
    spe_list = []
    accs_list = [0]*11
    sens_list = [0]*11
    spes_list = [0]*11

    for i in range(preds.shape[0]):

        auc, acc, sen, spe, accs, sens, spes = my_metrics(masks[i,:,:], preds[i,:,:], fovs[i,:,:])

        for j, elem in enumerate(accs):
            accs_list[j] += elem

        for j, elem in enumerate(sens):
            sens_list[j] += elem

        for j, elem in enumerate(spes):
            spes_list[j] += elem

        auc_list.append(auc)
        acc_list.append(acc)
        sen_list.append(sen)
        spe_list.append(spe)

    return np.mean(auc_list), np.mean(acc_list), np.mean(sen_list), np.mean(spe_list)

def my_metrics(y_true, y_pred, fov):

    fpr = []
    tpr = []
    accs = []
    sens = []
    spes = []

    for i in range(100,-1,-1):

        thresh = 0.01*i
        this_pred = np.zeros(y_pred.shape)
        this_pred[np.where(y_pred>=thresh)] = 1

        TP = np.sum(np.multiply(np.multiply(y_true, this_pred), fov))
        TN = np.sum(np.multiply(np.multiply(1-y_true, 1-this_pred),fov))
        FP = np.sum(np.multiply(np.multiply(1-y_true, this_pred), fov))
        FN = np.sum(np.multiply(np.multiply(y_true, 1-this_pred), fov))
        
        fpr.append(FP/(TN+FP))
        tpr.append(TP/(TP+FN))

        if i==50:
            acc = (TP+TN) / (TP+TN+FP+FN)
            sen = TP / (TP+FN)
            spe = TN / (TN+FP)

        if i%10==0:
            accs.append((TP+TN) / (TP+TN+FP+FN))
            sens.append(TP / (TP+FN))
            spes.append(TN / (TN+FP))

    fpr.append(1)
    tpr.append(1)

    return auc(fpr, tpr), acc, sen, spe, accs, sens, spes


def get_generator(train_x, train_y, train_sel, patch_size, batch_size, seed):

    rnd_state = np.random.RandomState(seed)
    safe_size = int(np.ceil(np.sqrt(2)*patch_size))

    if np.mod(safe_size,2) == 0:
        safe_size += 1

    l = safe_size//2

    batch_features = np.zeros((batch_size, 1, patch_size, patch_size))
    batch_labels   = np.zeros((batch_size, 1, patch_size, patch_size))

    while True:

        for i in range(batch_size):
            img_idx = rnd_state.randint(0, train_x.shape[0], 1)[0]
            img = train_x[img_idx,:,:]
            
            img_copy = deepcopy(img)
            mask = train_y[img_idx,:,:]
            
            mask_copy = deepcopy(mask)

            sel = train_sel[img_idx]
            sel_idx = rnd_state.randint(0,len(sel),1)         
            pos = sel[sel_idx[0]]
            
            safe_patch = img_copy[pos[0]-l:pos[0]+l+1,pos[1]-l:pos[1]+l+1]
            safe_mask  = mask_copy[pos[0]-l:pos[0]+l+1,pos[1]-l:pos[1]+l+1]
            
            # apply random perturbation
            if rnd_state.random_sample() > 0.5:                
                safe_patch = np.fliplr(safe_patch)
                safe_mask  = np.fliplr(safe_mask)
        
            else:                
                safe_patch = np.flipud(safe_patch)
                safe_mask  = np.flipud(safe_mask)

            rot_angle = rnd_state.randint(-9, 10) * 10
            safe_patch = rotate(safe_patch, rot_angle, reshape=False)
            safe_mask = rotate(safe_mask, rot_angle, reshape=False)            

            # keep center region of the patch
            c = safe_size//2
            l2 = patch_size//2
            patch_int = safe_patch[c-l2:c+l2, c-l2:c+l2]
            patch_int[np.where(patch_int>1)]=1
            patch_int[np.where(patch_int<0)]=0

            rr = rnd_state.random_sample()
            bias = rr*(1-np.max(patch_int))+(1-rr)*(-np.min(patch_int))
            patch_int = patch_int + bias
            mask_int = safe_mask[c-l2:c+l2, c-l2:c+l2]
            mask_int = mask_int >= 0.5

            batch_features[i,0,:,:] = patch_int
            batch_labels[i,0,:,:]   = mask_int

        yield batch_features, batch_labels


def get_mix_coef(init_alpha, decay, start_epoch, min_alpha, epoch):

    if epoch <= start_epoch:
        return init_alpha

    else:
        alpha = init_alpha - (epoch-start_epoch)*decay

        if alpha < min_alpha:
            return min_alpha
        else:
            return alpha


def log_gaussian(x, mu, logvar):

    PI = mu.new([np.pi])

    x = x.view(x.shape[0],-1)
    mu = mu.view(x.shape[0],-1)
    logvar = logvar.view(x.shape[0],-1)
    
    N, D = x.shape

    log_norm = (-1/2) * (D * torch.log(2*PI) + 
                         logvar.sum(dim=1) +
                         (((x-mu)**2)/(logvar.exp())).sum(dim=1))

    return log_norm



def topo_metric(gt, pred, thresh, n_paths):

    # 0, 1 and 2 mean, respectively, that path is infeasible, shorter/larger and correct
    result = []

    # binarize pred according to thresh
    pred_bw = (pred>thresh).astype(int)
    pred_cc = morphology.label(pred_bw)

    # get centerlines of gt and pred
    gt_cent = morphology.skeletonize(gt>0.5)
    gt_cent_cc = morphology.label(gt_cent)
    pred_cent = morphology.skeletonize(pred_bw)
    pred_cent_cc = morphology.label(pred_cent)

    # costs matrices
    gt_cost = np.ones(gt_cent.shape)
    gt_cost[gt_cent==0] = 10000
    pred_cost = np.ones(pred_cent.shape)
    pred_cost[pred_cent==0] = 10000
    
    # build graph and find shortest paths
    for i in range(n_paths):

        # pick randomly a first point in the centerline
        R_gt_cent, C_gt_cent = np.where(gt_cent==1)
        idx1 = randint(0, len(R_gt_cent)-1)
        label = gt_cent_cc[R_gt_cent[idx1], C_gt_cent[idx1]]
        ptx1 = (R_gt_cent[idx1], C_gt_cent[idx1])

        # pick a second point that is connected to the first one
        R_gt_cent_label, C_gt_cent_label = np.where(gt_cent_cc==label)
        idx2 = randint(0, len(R_gt_cent_label)-1)
        ptx2 = (R_gt_cent_label[idx2], C_gt_cent_label[idx2])

        # if points have different labels in pred image, no path is feasible
        if (pred_cc[ptx1] != pred_cc[ptx2]) or pred_cc[ptx1]==0:
            result.append(0)

        else:
            # find corresponding centerline points in pred centerlines
            R_pred_cent, C_pred_cent = np.where(pred_cent==1)
            poss_corr = np.zeros((len(R_pred_cent),2))
            poss_corr[:,0] = R_pred_cent
            poss_corr[:,1] = C_pred_cent
            poss_corr = np.transpose(np.asarray([R_pred_cent, C_pred_cent]))
            dist2_ptx1 = np.sum((poss_corr-np.asarray(ptx1))**2, axis=1)
            dist2_ptx2 = np.sum((poss_corr-np.asarray(ptx2))**2, axis=1)
            corr1 = poss_corr[np.argmin(dist2_ptx1)]
            corr2 = poss_corr[np.argmin(dist2_ptx2)]
            
            # find shortest path in gt and pred
            
            gt_path, cost1 = graph.route_through_array(gt_cost, ptx1, ptx2)
            gt_path = np.asarray(gt_path)

            pred_path, cost2 = graph.route_through_array(pred_cost, corr1, corr2)
            pred_path = np.asarray(pred_path)


            # compare paths length
            path_gt_length = np.sum(np.sqrt(np.sum(np.diff(gt_path, axis=0)**2, axis=1)))
            path_pred_length = np.sum(np.sqrt(np.sum(np.diff(pred_path, axis=0)**2, axis=1)))
            if pred_path.shape[0]<2:
                result.append(2)
            else:
                if ((path_gt_length / path_pred_length) < 0.9) or ((path_gt_length / path_pred_length) > 1.1):
                    result.append(1)
                else:
                    result.append(2)

    return result.count(0), result.count(1), result.count(2)
