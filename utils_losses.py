import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import utils_data

# ============================================
# ============================================
def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i]['count'] = 0
        bin_dict[i]['conf'] = 0
        bin_dict[i]['acc'] = 0
        bin_dict[i]['bin_acc'] = 0
        bin_dict[i]['bin_conf'] = 0

# ============================================
# ============================================
def _populate_bins(confs,
                   preds,
                   labels,
                   num_bins=10):
    
    bin_dict = {}
    
    for i in range(num_bins):
        bin_dict[i] = {}
    
    _bin_initializer(bin_dict, num_bins)
    
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
    
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
    
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn]['count'] = bin_dict[binn]['count'] + 1
        bin_dict[binn]['conf'] = bin_dict[binn]['conf'] + confidence
        bin_dict[binn]['acc'] = bin_dict[binn]['acc'] + (1 if (label == 1 and prediction == 1) else 0)
        # bin_dict[binn]['acc'] = bin_dict[binn]['acc'] + (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
    
        if (bin_dict[binn]['count'] == 0):
            bin_dict[binn]['bin_acc'] = 0
            bin_dict[binn]['bin_conf'] = 0
    
        else:
            bin_dict[binn]['bin_acc'] = float(bin_dict[binn]['acc']) / bin_dict[binn]['count']
            bin_dict[binn]['bin_conf'] = bin_dict[binn]['conf'] / float(bin_dict[binn]['count'])
    
    return bin_dict

# ============================================
# ============================================
def compute_calibration_error(confs,
                              preds,
                              labels,
                              num_bins = 10,
                              aggr = 'expected'):
    
    bin_dict = _populate_bins(confs,
                              preds,
                              labels,
                              num_bins)
    
    num_samples = len(labels)
    non_empty_bins = 0
    ce = 0
    ce_list = []

    for i in range(num_bins):

        bin_accuracy = bin_dict[i]['bin_acc']
        bin_confidence = bin_dict[i]['bin_conf']
        bin_count = bin_dict[i]['count']

        if aggr == 'expected':
            ce += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)

        elif aggr == 'average': 
            if bin_count > 0: non_empty_bins += 1
            ce += abs(bin_accuracy - bin_confidence)   

        elif aggr == 'maximum':
            ce_list.append(abs(bin_accuracy - bin_confidence))

        elif aggr == 'l2':
            ce += (float(bin_count) / num_samples) * (bin_accuracy - bin_confidence)**2    

    if aggr == 'average': 
        ce = ce / float(non_empty_bins)
    elif aggr == 'maximum': 
        ce = max(ce_list)
    elif aggr == 'l2': 
        ce = math.sqrt(ce)

    return ce

# https://github.com/mobarakol/SVLS/blob/main/calibration_metrics.py
def ece_eval(preds,
             targets,
             n_bins=10,
             bg_cls=0):
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    # Mobarakol's implementation only considers pixels that are labelled as foreground in the ground truth labels
    confidences, predictions = confidences[targets > bg_cls], predictions[targets > bg_cls]
    accuracies = (predictions == targets[targets > bg_cls])
    
    Bm, acc, conf = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    ece = 0.0
    bin_idx = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        Bm[bin_idx] = bin_size
        
        if bin_size > 0:  
    
            accuracy_in_bin = np.sum(accuracies[in_bin])
            acc[bin_idx] = accuracy_in_bin / Bm[bin_idx]
            confidence_in_bin = np.sum(confidences[in_bin])
            conf[bin_idx] = confidence_in_bin / Bm[bin_idx]
        
            bin_idx += 1
        
    ece_all = Bm * np.abs((acc - conf)) / Bm.sum()
    
    ece = ece_all.sum() 
    
    return ece, acc, conf, Bm

# ==========================================
# returns a tensor of the same size as the input probs (bs, num_classes, x, y)
# ==========================================  
def get_kl(probs1,
           probs2,
           eps = 1e-12):
    
    kl = probs1 * (torch.log(probs1 + eps) - torch.log(probs2 + eps))
    # kl = torch.sum(kl, dim = 1) # NOT summing across classes here
    return kl
    
# ==========================================  
# returns a tensor of the same size as the input probs (bs, num_classes, x, y)
# ==========================================  
def get_js(probs1,
           probs2,
           lam = 0.5):
    
    probs_average = lam * probs1 + (1 - lam) * probs2

    return lam * get_kl(probs1, probs_average) + (1 - lam) * get_kl(probs2, probs_average)

# ==========================================
# ==========================================    
def get_losses(preds,
               targets,
               mask = None,
               loss_type = 'ce',
               margin = 0.1):
        
    if loss_type == 'ce':
        loss_pc = - targets * F.log_softmax(preds, dim=1) # pixel wise cross entropy
    elif loss_type == 'l2':
        loss_pc = torch.square(preds - targets) # pixel wise square difference
    elif loss_type == 'l2_all':
        loss_pc = torch.square(preds - targets) # pixel wise square difference (using another tag to indicate consistency at all layers)
    elif loss_type == 'l2_margin': # pixel wise square difference | thresholded at margin
        loss_pc = torch.maximum(torch.square(preds - targets), margin * torch.ones_like(torch.square(preds - targets)))
    elif loss_type == 'js': # pixel wise square difference | thresholded at margin
        loss_pc = get_js(preds, targets)

    if mask != None:
        loss_pc = torch.mul(loss_pc, mask)
    
    loss_p = torch.mean(loss_pc, dim = 1) # average across channels (classes in the last layer)
    loss = torch.mean(loss_p) # mean over all pixels and all images in the batch
    
    return loss_pc, loss_p, loss
    
# ==========================================
# the weights here are for relative weighting of pixels within the consistency loss only
# the importance of consistency loss vs supervised loss is set by the args.l2 parameter when the losses are combined
# ==========================================
def get_lambda_maps(labels,
                    weigh_per_distance, # whether to make weight vary as per distance from the boundary or not
                    num_labels,
                    lmax = 1.0, # max value (applied to pixels on the tissue boundary)
                    R = 10.0, # margin around the boundary where the lambda values vary
                    drop_factor = 100.0, # min value (applied to pixels farther away than the margin) = lmax / drop_factor
                    alp = 1.0): # rate of decrease of lambda away from the boundary
                     
    weights = lmax * np.ones_like(labels, dtype = np.float32)
    
    if weigh_per_distance == 1:

        for idx in range(labels.shape[0]):

            if num_labels == 2: # binary segmentations
                sdt = utils_data.compute_sdt(labels[idx, 0, :, :])
                weights[idx, 0, :, :] = utils_data.compute_lambda_map(sdt, R, lmax / drop_factor, lmax, alp)

            else: # multi-label segmentations
                label = labels[idx, 0, :, :]
                lamdas = []
                
                for c in range(1, num_labels):
                    
                    label_tmp = np.copy(label)
                    label_tmp[label_tmp != c] = 0
                    label_tmp[label_tmp == c] = 1
                    
                    sdt = utils_data.compute_sdt(label_tmp)
                    lam = utils_data.compute_lambda_map(sdt, R, lmax / drop_factor, lmax, alp)
                    lamdas.append(lam)  

                weights[idx, 0, :, :] = np.max(np.array(lamdas), axis=0)
        
    return weights

# ==========================================
# inspired from https://github.com/mobarakol/SVLS/blob/main/svls.py
# ==========================================
def smoothen_labels(labels,
                    num_labels,
                    device,
                    kernel_size = 3,
                    sigma = 1.0):

    svls_filter_2d = torch.nn.Conv2d(in_channels = num_labels,
                                     out_channels = num_labels,
                                     kernel_size = kernel_size,
                                     groups = num_labels,
                                     bias = False,
                                     padding = 0)

    svls_kernel_2d = get_svls_filter_2d(num_classes = num_labels,
                                        kernel_size = kernel_size,
                                        sigma = sigma)
    svls_kernel_2d = svls_kernel_2d.to(device, dtype = torch.float)
       
    svls_filter_2d.weight.data = svls_kernel_2d
    svls_filter_2d.weight.requires_grad = False 

    labels_1hot = utils_data.make_label_onehot(labels, num_labels)
    labels_1hot = F.pad(labels_1hot, (1, 1, 1, 1), mode='replicate')
    labels_svls = svls_filter_2d(labels_1hot) / svls_kernel_2d[0].sum()

    return labels_svls

# ==========================================
# copied from https://github.com/mobarakol/SVLS/blob/main/svls.py
# ==========================================
def get_svls_filter_2d(num_classes,
                       kernel_size = 3,
                       sigma = 1):
    
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 3)
    x_coord = torch.arange(kernel_size)
    x_grid_2d = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)    
    y_grid_2d = x_grid_2d.t()
    xy_grid_2d = torch.stack([x_grid_2d, y_grid_2d], dim=-1).float()
    
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    # Calculate the 2-dimensional gaussian kernel
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp(
                            -torch.sum((xy_grid_2d - mean)**2., dim=-1) / (2*variance + 1e-16))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    neighbors_sum = 1 - gaussian_kernel[1,1]
    gaussian_kernel[1,1] = neighbors_sum
    svls_kernel_2d = gaussian_kernel / neighbors_sum

    # Reshape to 2d depthwise convolutional weight
    svls_kernel_2d = svls_kernel_2d.view(1, 1, kernel_size, kernel_size)
    svls_kernel_2d = svls_kernel_2d.repeat(num_classes, 1, 1, 1)
    
    return svls_kernel_2d