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
import utils_losses
import data_loader
import csv
# eval metrics
from medpy.metric.binary import assd as ASSD
from medpy.metric.binary import hd as Hausdorff_Distance
from medpy.metric.binary import hd95 as Hausdorff_Distance_95

# ===================================================
# Function to compute metrics
# ===================================================
def compute_metrics(pred,
                    label,
                    voxelspacing = 1.0):

    scores = np.zeros(5, dtype = np.float32) 
    
    # dice    
    d = utils_generic.dice(im1 = pred, im2 = label)

    # hd, hd95 and assd
    if np.sum(pred) > 0:
        hd = Hausdorff_Distance(result = pred, reference = label, voxelspacing = voxelspacing)
        hd95 = Hausdorff_Distance_95(result = pred, reference = label, voxelspacing = voxelspacing)
        assd = ASSD(result = pred, reference = label, voxelspacing = voxelspacing)
    else:
        hd = np.nan
        hd95 = np.nan
        assd = np.nan

    scores[0] = d    
    scores[1] = hd
    scores[2] = hd95
    scores[3] = assd

    return scores

# ===================================================
# function to write obtained results to file
# ===================================================
def write_results_to_file(results_path,
                          test_subdataset,  
                          subject_metrics,
                          subject_names_ts,
                          num_decimals = 4,
                          metric = 'dice'):
    
    with open(results_path + '/' + test_subdataset + '_' + metric + '.csv', mode='w') as csv_file:
        
        csv_file = csv.writer(csv_file, delimiter=',')
        
        csv_file.writerow(['subject', metric])
        csv_file.writerow(['--------', '--------'])
        
        # for each subject
        for s in range(len(subject_names_ts)):
            csv_file.writerow([subject_names_ts[s], np.round(subject_metrics[s], num_decimals)])
        csv_file.writerow(['--------', '--------'])
        
        # statistics
        subject_metrics_array = np.array(subject_metrics)
        csv_file.writerow(['Mean', np.round(np.mean(subject_metrics_array), num_decimals)])
        csv_file.writerow(['Standard deviation', np.round(np.std(subject_metrics_array), num_decimals)])
        csv_file.writerow(['--------', '--------'])

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
    
    parser.add_argument('--dataset', default='prostate') # placenta / prostate
    parser.add_argument('--sub_dataset', default='RUNMC') # prostate: BIDMC / BMC / HK / I2CVB / RUNMC / UCL
    parser.add_argument('--test_sub_dataset', default='RUNMC') # prostate: BIDMC / BMC / HK / I2CVB / RUNMC / UCL
    parser.add_argument('--cv_fold_num', default=1, type=int)
    parser.add_argument('--num_labels', default=2, type=int)

    parser.add_argument('--save_path', default='/data/scratch/nkarani/projects/crael/seg/logdir/v8/')
    
    parser.add_argument('--data_aug_prob', default=0.5, type=float)
    parser.add_argument('--optimizer', default='adam') # adam / sgd
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    
    # no tricks: (100), data aug (010), data aug + consistency (011 / 012)
    parser.add_argument('--l0', default=0.0, type=float) # 0 / 1
    parser.add_argument('--l1', default=1.0, type=float) # 0 / 1
    parser.add_argument('--l2', default=1.0, type=float) # 0 / 1 
    parser.add_argument('--l1_loss', default='ce') # 'ce' / 'dice'
    parser.add_argument('--l2_loss', default='l2_all') # 'ce' / 'l2' / 'l2_margin' 
    parser.add_argument('--l2_loss_margin', default=0.1, type=float) # 0.1
    parser.add_argument('--alpha_layer', default=10.0, type=float) # 1.0
    parser.add_argument('--temp', default=1.0, type=float) # 1 / 2
    parser.add_argument('--teacher', default='self') # 'self' / 'ema'
    
    parser.add_argument('--run_number', default=1, type=int)
    parser.add_argument('--debugging', default=0, type=int)    
    
    args = parser.parse_args()

    # ===================================
    # set random seed
    # ===================================
    logging.info('Setting random seeds for numpy and torch')
    np.random.seed(args.run_number)
    torch.manual_seed(args.run_number)

    # ===================================
    # select device - gpu / cpu
    # ===================================
    logging.info('Finding out which device I am running on')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # ===================================
    # directories
    # ===================================
    logging_dir = utils_generic.make_expdir_2(args)
    models_path = logging_dir + 'models/'
    results_path = logging_dir + 'results/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # ===================================
    # define model
    # ===================================
    logging.info('Defining segmentation model')
    model = models.UNet2d(in_channels = 1, num_labels = args.num_labels, squeeze = False)
    model = model.to(device)

    # ===================================
    # load model weights
    # ===================================
    modelpath = utils_generic.get_best_modelpath(models_path, 'best_ema_val_iter')
    logging.info('loading model weights from: ')
    logging.info(modelpath)
    model.load_state_dict(torch.load(modelpath)['state_dict'])

    # ===================================
    # Set model to eval mode
    # ===================================
    model.eval()

    # ===================================
    # load test images and labels
    # ===================================
    logging.info('Evaluation experiment: ' + logging_dir)
    logging.info('Reading test data')
    data_test = data_loader.load_data(args.dataset, args.test_sub_dataset, args.cv_fold_num, 'test')
    images_ts = data_test["images"]
    labels_ts = data_test["labels"]
    depths_ts = data_test["depths"]
    subject_names_ts = data_test["subject_names"]

    # ===================================
    # evaluate each subject one by one
    # ===================================
    num_subjects_ts = subject_names_ts.shape[0]
    logging.info('number of test subjects: ' + str(num_subjects_ts))
    logging.info('depth dimensions of these test subjects: ' + str(depths_ts))
    subject_dices = []
    subject_eces = []
    
    images_all = np.zeros((num_subjects_ts, images_ts.shape[0], images_ts.shape[1]))
    labels_all = np.zeros((num_subjects_ts, images_ts.shape[0], images_ts.shape[1]))
    logits0_all = np.zeros((num_subjects_ts, images_ts.shape[0], images_ts.shape[1]))
    logits1_all = np.zeros((num_subjects_ts, images_ts.shape[0], images_ts.shape[1]))
    prob_fg_all = np.zeros((num_subjects_ts, images_ts.shape[0], images_ts.shape[1]))

    for sub in range(num_subjects_ts):
    
        # get subject's image and labels
        subject_name = subject_names_ts[sub]
        logging.info(subject_name)
        subject_image = images_ts[:,:,int(np.sum(depths_ts[:sub])):int(np.sum(depths_ts[:sub+1]))]
        subject_label = labels_ts[:,:,int(np.sum(depths_ts[:sub])):int(np.sum(depths_ts[:sub+1]))]

        # make prediction
        logits, probs = utils_data.predict_logits_and_probs(subject_image, model, device)
        soft_prediction = probs[:,1,:,:]

        # threshold probability
        hard_prediction = (soft_prediction > 0.5).astype(np.float32)

        # ===================
        # resolution scaling is needed for prostate datasets
        # ===================
        if args.dataset == 'prostate':

            # ===================
            # read original image and label (without preprocessing)
            # ===================
            image_orig, label_orig = data_loader.load_without_preproc(args.test_sub_dataset, subject_name.decode('utf-8'))
            label_orig[label_orig!=0] = 1

            if args.test_sub_dataset in ['UCL', 'HK', 'BIDMC']:
                label_orig = np.swapaxes(np.swapaxes(label_orig, 0, 1), 1, 2)
                image_orig = np.swapaxes(np.swapaxes(image_orig, 0, 1), 1, 2)

            # ===================
            # convert the predicitons back to original resolution
            # ===================
            hard_prediction_orig_res_and_size = utils_data.rescale_and_crop(hard_prediction,
                                                                            scale = (0.625 / data_test["px"][sub], 0.625 / data_test["py"][sub]),
                                                                            size = (data_test["nx"][sub], data_test["ny"][sub]),
                                                                            order = 0).astype(np.uint8)
            
            probs_fg_orig_res_and_size = utils_data.rescale_and_crop(probs[:,1,:,:],
                                                                     scale = (0.625 / data_test["px"][sub], 0.625 / data_test["py"][sub]),
                                                                     size = (data_test["nx"][sub], data_test["ny"][sub]),
                                                                     order = 1)
            probs_bg_orig_res_and_size = 1 - probs_fg_orig_res_and_size
            probs_orig_res_and_size = np.stack((probs_bg_orig_res_and_size, probs_fg_orig_res_and_size), axis = 1)

            # compute calibration error
            subject_eces.append(utils_losses.ece_eval(probs_orig_res_and_size, np.swapaxes(np.swapaxes(label_orig, 1, 2), 0, 1), n_bins=15)[0])

        elif args.dataset in ['placenta', 'ms']:
            hard_prediction_orig_res_and_size = hard_prediction
            image_orig = subject_image
            label_orig = subject_label

        # ===================
        # original images and labels are (x,y,z)
        # predictions are (z, x, y)
        # swap axes of predictions to fix this
        # ===================
        hard_prediction_orig_res_and_size = np.swapaxes(np.swapaxes(hard_prediction_orig_res_and_size, 0, 1), 1, 2)
        logging.info(hard_prediction_orig_res_and_size.shape)
        logging.info(label_orig.shape)

        # ===================
        # visualize results
        # ===================
        utils_vis.save_results(image_orig,
                               label_orig,
                               hard_prediction_orig_res_and_size,
                               results_path + args.test_sub_dataset + '_' + subject_name.decode('utf-8') + '.png')

        # ===================
        # compute dice
        # ===================
        subject_dices.append(compute_metrics(pred = hard_prediction_orig_res_and_size, label = label_orig)[0])

        # ===================
        # collect 2d slices with largest foreground
        # ===================
        idx = np.argmax(np.sum(subject_label, axis=(0,1)))
        images_all[sub, :, :] = subject_image[:, :, idx]
        labels_all[sub, :, :] = subject_label[:, :, idx]
        logits0_all[sub, :, :] = logits[idx, 0, :, :]
        logits1_all[sub, :, :] = logits[idx, 1, :, :]
        prob_fg_all[sub, :, :] = soft_prediction[idx, :, :]

    # ===================
    # visualize results
    # ===================
    utils_vis.save_all([images_all, labels_all, logits0_all, logits1_all, prob_fg_all],
                        results_path + args.test_sub_dataset + '_all.png',
                        'numpy',
                        cmaps = ['gray', 'gray', 'gray', 'gray', 'jet'])
        
    # ===================
    # write quantitative results
    # ===================
    write_results_to_file(results_path, args.test_sub_dataset, subject_dices, subject_names_ts, metric = 'dice')
    write_results_to_file(results_path, args.test_sub_dataset, subject_eces, subject_names_ts, metric = 'ece')