# ===================================
# import stuff
# ===================================
# book-keeping stuff
import os
import argparse
import logging
# cnn related stuff
import numpy as np
import torch
import torch.nn as nn
# self-defined stuff
import models 
import utils_data
import utils_vis
import utils_generic
import data_loader
import csv

# ==========================================
# ==========================================
def get_results(args):
    
    if args.dataset == 'prostate':
        test_subdatasets = ['BMC', 'UCL', 'HK', 'BIDMC']
        num_test_subjects = 10
        num_runs = 5
    elif args.dataset == 'ms':
        test_subdatasets = ['InD']
        num_test_subjects = 33
        num_runs = 3
    
    results = np.zeros((len(test_subdatasets) * num_test_subjects, num_runs))
    
    for r in range(num_runs):
    
        args.run_number = r+1
        logging_dir = utils_generic.make_expdir(args)
        results_path = logging_dir + 'results/'

        sub_num = 0

        for test_subdataset in test_subdatasets:
            
            results_filename = results_path + test_subdataset + '.csv'

            rownum = 0
            with open(results_filename, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ')
                for row in csvreader:
                    rownum = rownum + 1
                    if rownum > 2 and rownum < 3 + num_test_subjects:
                        results[sub_num, r] = row[0][row[0].find(',')+1:]
                        sub_num = sub_num + 1

    logging.info('=======================================')
    logging.info(args)
    logging.info(np.round(np.mean(results), 3))
    logging.info(np.round(np.std(results), 3))

    return results

# ==========================================
# Carries out a paired permutation test
# with differences between the two methods for each subject
# scaled down by the geometric mean of the standard deviations of each method across training runs.
# ==========================================
def compute_significance(r1, r2, s1, s2, cv):
    mu1 = np.mean(r1, axis=-1)
    mu2 = np.mean(r2, axis=-1)
    std1 = np.std(r1, axis=-1)
    std2 = np.std(r2, axis=-1)
    den = np.sqrt(std1 * std2)
    # hack to deal with zero std devs
    den[den == 0.0] = 100.0
    den_tmp = np.min(den)
    den[den == 100.0] = den_tmp
    # differences with high std devs across runs will have small weights
    diff = (mu2 - mu1) / den
    # diff = mu2 - mu1
    # logging.info(np.mean(diff))
    # logging.info(np.std(diff))

    np.random.seed(1234)
    
    orig_diff = np.mean(diff)
    num_permutes = 10**4
    permute_diffs = 1234 * np.ones((num_permutes))
    for n in range(num_permutes):
        tmp = np.random.choice([-1,1], size=mu1.shape[0], replace=True)
        permute_diffs[n] = np.mean(tmp * diff)
    
    if orig_diff > 0.0:
        num = len(np.where(permute_diffs > orig_diff)[0]) + 1
    elif orig_diff < 0.0:
        num = len(np.where(permute_diffs < orig_diff)[0]) + 1
    p = num / (num_permutes + 1)

    logging.info('================= ' + str(cv))
    logging.info('Mean dice ' + s1 + ': ' + str(np.round(np.mean(r1), 4)))
    logging.info('Mean dice ' + s2 + ': ' + str(np.round(np.mean(r2), 4)))
    logging.info('Mean difference: ' + str(np.round(orig_diff, 4)) + ', p-value: ' + str(np.round(p, 2)))

    return 0

# ==========================================
# ==========================================
if __name__ == "__main__":

    # ===================================
    # setup logging
    # ===================================
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    
    # ===================================
    # read arguments
    # ===================================
    logging.info('Parsing arguments')

    parser = argparse.ArgumentParser(description = 'train segmentation model')
    
    parser.add_argument('--dataset', default='ms') # placenta / prostate / ms
    parser.add_argument('--sub_dataset', default='RUNMC') # prostate: BIDMC / BMC / HK / I2CVB / RUNMC / UCL
    parser.add_argument('--test_sub_dataset', default='RUNMC') # prostate: BIDMC / BMC / HK / I2CVB / RUNMC / UCL
    parser.add_argument('--cv_fold_num', default=1, type=int)
    parser.add_argument('--num_labels', default=2, type=int)

    parser.add_argument('--save_path', default='/data/scratch/nkarani/projects/crael/seg/logdir/v3/')
    
    parser.add_argument('--data_aug_prob', default=0.5, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_test', default=4, type=int)
    
    parser.add_argument('--model_has_heads', default=0, type=int)    
    parser.add_argument('--method_invariance', default=100, type=int) # 0: no reg, 1: data aug, 2: consistency, 3: consistency in each layer
    parser.add_argument('--lambda_dataaug', default=1.0, type=float) # weight for data augmentation loss
    parser.add_argument('--consis_loss', default=1, type=int) # 1: MSE | 2: MSE of normalized images (BYOL)
    parser.add_argument('--lambda_consis', default=0.1, type=float) # weight for regularization loss (consistency overall)
    parser.add_argument('--alpha_layer', default=100.0, type=float) # growth of regularization loss weight with network depth
    
    parser.add_argument('--run_number', default=1, type=int)
    parser.add_argument('--debugging', default=0, type=int)    
    
    args = parser.parse_args()

    for cv in [1]:
        args.cv_fold_num = cv

        # set method and relevant parameters in args
        # call function, passing args that gives you results of that method for all 40 subjects for all three runs in a (40,3) array
        args.method_invariance = 0
        args.model_has_heads = 0
        results_m0 = get_results(args)

        args.method_invariance = 100
        args.model_has_heads = 0
        results_m100 = get_results(args)

        args.method_invariance = 200
        args.model_has_heads = 1
        args.lambda_consis = 0.001
        args.alpha_layer = 10.0
        args.consis_loss = 2
        results_m200_lam001_alpha10 = get_results(args)

        args.method_invariance = 200
        args.model_has_heads = 1
        args.lambda_consis = 0.01
        args.alpha_layer = 10.0
        args.consis_loss = 2
        results_m200_lam01_alpha10 = get_results(args)

        args.method_invariance = 200
        args.model_has_heads = 1
        args.lambda_consis = 0.001
        args.alpha_layer = 100.0
        args.consis_loss = 2
        results_m200_lam001_alpha100 = get_results(args)

        args.method_invariance = 200
        args.model_has_heads = 1
        args.lambda_consis = 0.01
        args.alpha_layer = 100.0
        args.consis_loss = 2
        results_m200_lam01_alpha100 = get_results(args)

        compute_significance(results_m100, results_m200_lam001_alpha10, 'data aug', 'CR, lambda 0.001, alpha 10.0', cv)
        compute_significance(results_m100, results_m200_lam01_alpha10, 'data aug', 'CR, lambda 0.01, alpha 10.0', cv)
        compute_significance(results_m100, results_m200_lam001_alpha100, 'data aug', 'CR, lambda 0.001, alpha 100.0', cv)
        compute_significance(results_m100, results_m200_lam01_alpha100, 'data aug', 'CR, lambda 0.01, alpha 100.0', cv)