import numpy as np
import utils_vis
import utils_data
import itertools
import logging
import significance_tests
import data_loader
import argparse
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ============================================================================
# ============================================================================
plot_scatter = False
if plot_scatter == True:
    for cv in [1, 2]:
        for s in ['m100_da0.5_lda1.0', 'm200_da0.5_lda1.0_lcon0.01_a100.0_l2']:
            for measure_var in ['full', 'end']:
                a = []
                b = []
                c = []
                for r in [1, 2, 3]:
                    bpath = '/data/scratch/nkarani/projects/crael/seg/logdir/v6/prostate_RUNMC_cv' + str(cv) + '/lr0.0001_sch2_bs16/unet/'
                    mpath = bpath + s + '/run' + str(r) + '/results/'
                    a.append(np.load(mpath + 'ind_val_' + measure_var + '.npy'))
                    b.append(np.load(mpath + 'ind_test_' + measure_var + '.npy'))
                    c.append(np.load(mpath + 'ood_test_' + measure_var + '.npy'))

                utils_vis.plot_scatter_simple(a, b, c, bpath+s, measure_var)

# ============================================================================
# ============================================================================
def get_odd_test_scores(s, random_subs):

    test_dice = []
    for r in [1, 2, 3]:
        
        mpath = bpath + s + '/run' + str(r) + '/results/'

        # select best model according to val set consisting of the given set of subjects
        best_model = np.argmax(np.mean(np.load(mpath + 'ind_test_' + measure_var + '.npy')[:,random_subs], -1))

        test_dice.append(np.load(mpath + 'ood_test_' + measure_var + '.npy')[best_model, :])

    return np.array(test_dice), best_model

# ============================================================================
# ============================================================================
analyze_model_selection_variance = True
if analyze_model_selection_variance == True:
    
    for cv in [2]:
        
        bpath = '/data/scratch/nkarani/projects/crael/seg/logdir/v6/prostate_RUNMC_cv' + str(cv) + '/lr0.0001_sch2_bs16/unet/'

        # number of different combinations of validation subjects
        # total number of val subjects is 5, let's consider all combinations of 3 subjects
        combinations = list(itertools.combinations(range(10), 3))
        measure_var = 'full'
        s1 = 'm100_da0.5_lda1.0'
        s2 = 'm200_da0.5_lda1.0_lcon0.01_a100.0_l2'
        # s1 = 'm0'
        s1better = 0
        s2better = 0
        best_models1 = []
        best_models2 = []
        for subjects in combinations:
            r1, best_model1 = np.transpose(get_odd_test_scores(s1, subjects))
            r2, best_model2 = np.transpose(get_odd_test_scores(s2, subjects))
            best_models1.append(best_model1)
            best_models2.append(best_model2)
            diff, p = significance_tests.compute_significance(r1, r2, s1, s2, cv, False)

            if diff > 0.0 and p < 0.05:
                s2better = s2better + 1
                # logging.info(str(best_model1) + ", " + str(best_model2) + " as m1, m2 lead to m2 being better.")
            elif diff < 0.0 and p < 0.05:
                s1better = s1better + 1
                # logging.info(str(best_model1) + ", " + str(best_model2) + " as m1, m2 lead to m1 being better.")

        for b in np.unique(np.array(best_models1)):
            logging.info("for method 1, model " + str(b) + " was selected " + str(len(np.where(np.array(best_models1) == b)[0])) + " times.")
        for b in np.unique(np.array(best_models2)):
            logging.info("for method 2, model " + str(b) + " was selected " + str(len(np.where(np.array(best_models2) == b)[0])) + " times.")
        logging.info("========= In cv " + str(cv) + ", out of " + str(len(combinations)) + " validation sets, ")
        logging.info("        for " + str(s2better) + " times, Method 2 is better with p < 0.05")
        logging.info("        for " + str(s1better) + " times, Method 1 is better with p < 0.05")
        logging.info("        for " + str(len(combinations) - s1better - s2better) + " times, there is no statistically significant diff between the methods.")

# ============================================================================
# ============================================================================
plot_val_ood_corr = False
if plot_val_ood_corr == True:
    cv = 2
    bpath = '/data/scratch/nkarani/projects/crael/seg/logdir/v6/prostate_RUNMC_cv' + str(cv) + '/lr0.0001_sch2_bs16/unet/'
    measure_var = 'full' # consider models equally spaced at 10k iterations from 10k to 100k
    s = 'm100_da0.5_lda1.0' # 'm200_da0.5_lda1.0_lcon0.01_a100.0_l2' # 'm200_da0.5_lda1.0_lcon0.01_a100.0_l2' / m100_da0.5_lda1.0
    for r in [1, 2, 3]:

        # path for this setting
        mpath = bpath + s + '/run' + str(r) + '/results/'

        # read ind val and ood test scores    
        ind_val = np.load(mpath + 'ind_val_' + measure_var + '.npy')
        ind_tst = np.load(mpath + 'ind_test_' + measure_var + '.npy')
        ood_tst = np.load(mpath + 'ood_test_' + measure_var + '.npy')

        utils_vis.plot_ind_ood_corr(ind_val, ind_tst, ood_tst, mpath + 'ind_ood_corr')

# ============================================================================
# ============================================================================
plot_val_ood_corr2 = False
if plot_val_ood_corr2 == True:
    cv = 2
    bpath = '/data/scratch/nkarani/projects/crael/seg/logdir/v6/prostate_RUNMC_cv' + str(cv) + '/lr0.0001_sch2_bs16/unet/'
    measure_var = 'full' # consider models equally spaced at 10k iterations from 10k to 100k
    s = 'm100_da0.5_lda1.0' # 'm200_da0.5_lda1.0_lcon0.01_a100.0_l2' / m100_da0.5_lda1.0
    ind_val = []
    ind_tst = []
    ood_tst = []
    for r in [1, 2, 3]:

        # path for this setting
        mpath = bpath + s + '/run' + str(r) + '/results/'

        # read ind val and ood test scores    
        ind_val.append(np.load(mpath + 'ind_val_' + measure_var + '.npy'))
        ind_tst.append(np.load(mpath + 'ind_test_' + measure_var + '.npy'))
        ood_tst.append(np.load(mpath + 'ood_test_' + measure_var + '.npy'))

    ind_val = np.array(ind_val)
    ind_tst = np.array(ind_tst)
    ood_tst = np.array(ood_tst)

    utils_vis.plot_ind_ood_corr2(ind_val, ind_tst, ood_tst, bpath + s + '/ind_ood_scatter')

# ============================================================================
# ============================================================================
def save_some_images(args, subdataset, ttv, augmented = False):
    
    bpath = '/data/scratch/nkarani/projects/crael/seg/logdir/v6/example_images/'
    data = data_loader.load_data(args, subdataset, ttv)

    if augmented == False:        
        for sub in range(4):
            image = data['images'][:,:,int(np.sum(data['depths'][:sub])):int(np.sum(data['depths'][:sub+1]))]
            utils_vis.save_image(image[:, :, image.shape[-1]//2], bpath + subdataset + data["subject_names"][sub].decode('utf-8') + '.png')

    elif augmented == True:

        sub = 0
        image = data['images'][:,:,int(np.sum(data['depths'][:sub])):int(np.sum(data['depths'][:sub+1]))]
        inputs_cpu, labels_cpu = utils_data.get_batch(data['images'], data['labels'], 2, batch_type = 'sequential', start_idx = image.shape[-1]//2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for aug in range(20):
            images_aug, _, _ = utils_data.transform_batch(utils_data.torch_and_send_to_device(inputs_cpu, device),
                                                          utils_data.torch_and_send_to_device(labels_cpu, device),
                                                          data_aug_prob = args.data_aug_prob,
                                                          device = device)
            
            utils_vis.save_image(images_aug.detach().cpu().numpy()[0,0,:,:], bpath + subdataset + data["subject_names"][sub].decode('utf-8') + '_aug' + str(aug) + '.png')

# ============================================================================
# ============================================================================
vis_example_images = False
if vis_example_images == True:

    parser = argparse.ArgumentParser()  
    parser.add_argument('--dataset', default='prostate') # placenta / prostate
    parser.add_argument('--cv_fold_num', default=1, type=int)
    parser.add_argument('--data_aug_prob', default=0.5, type=float)

    args = parser.parse_args()

    save_some_images(args, 'RUNMC', 'test')
    save_some_images(args, 'BMC', 'test')
    save_some_images(args, 'UCL', 'test')
    save_some_images(args, 'HK', 'test')
    save_some_images(args, 'BIDMC', 'test')

# ============================================================================
# ============================================================================
vis_example_augmented_images = False
if vis_example_augmented_images == True:

    parser = argparse.ArgumentParser()  
    parser.add_argument('--dataset', default='prostate') # placenta / prostate
    parser.add_argument('--cv_fold_num', default=1, type=int)
    parser.add_argument('--data_aug_prob', default=0.5, type=float)

    args = parser.parse_args()

    save_some_images(args, 'RUNMC', 'test', augmented = True)
