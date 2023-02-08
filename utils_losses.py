import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


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