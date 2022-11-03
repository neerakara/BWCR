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
# eval metrics
from medpy.metric.binary import assd as ASSD
from medpy.metric.binary import hd as Hausdorff_Distance
from medpy.metric.binary import hd95 as Hausdorff_Distance_95

# ======================================================
# ======================================================
def dice(im1,
         im2,
         empty_score=1.0):
    
    im1 = im1 > 0.5
    im2 = im2 > 0.5
    
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

# ===================================================
# Function to compute metrics
# ===================================================
def compute_metrics(pred,
                    label,
                    voxelspacing = 1.0):

    scores = np.zeros(5, dtype = np.float32) 
    
    # dice    
    d = dice(im1 = pred, im2 = label)

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
                          subject_dices,
                          subject_names_ts,
                          num_decimals = 4):
    
    with open(results_path + '/' + test_subdataset + '.csv', mode='w') as csv_file:
        
        csv_file = csv.writer(csv_file, delimiter=',')
        csv_file.writerow(['subject', 'dice'])
        csv_file.writerow(['--------', '--------'])
        # for each subject
        for s in range(len(subject_names_ts)):
            csv_file.writerow([subject_names_ts[s], np.round(subject_dices[s], num_decimals)])
        csv_file.writerow(['--------', '--------'])
        subject_dices_array = np.array(subject_dices)
        csv_file.writerow(['Mean', np.round(np.mean(subject_dices_array), num_decimals)])
        csv_file.writerow(['Standard deviation', np.round(np.std(subject_dices_array), num_decimals)])
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

    parser.add_argument('--save_path', default='/data/scratch/nkarani/projects/crael/seg/logdir/')
    
    parser.add_argument('--data_aug_prob', default=0.5, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_test', default=8, type=int)
    
    parser.add_argument('--model_has_heads', default=0, type=int)    
    parser.add_argument('--method_invariance', default=0, type=int) # 0: no reg, 1: data aug, 2: consistency, 3: consistency in each layer
    parser.add_argument('--lambda_data_aug', default=1.0, type=float) # weight for regularization loss (data augmentation)
    parser.add_argument('--lambda_consis', default=1.0, type=float) # weight for regularization loss (consistency overall)
    parser.add_argument('--alpha_layer', default=1.0, type=float) # growth of regularization loss weight with network depth
    
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
    logging_dir = utils_generic.make_expdir(args)
    models_path = logging_dir + 'models/'
    results_path = logging_dir + 'results/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # ===================================
    # load test images and labels
    # ===================================
    logging.info('Reading test data')
    data_test = data_loader.load_data(args, args.test_sub_dataset, 'test')
    images_ts = data_test['images']
    labels_ts = data_test['labels']
    depths_ts = data_test['depths']
    subject_names_ts = data_test['subject_names']

    # ===================================
    # define model
    # ===================================
    logging.info('Defining segmentation model')
    if args.model_has_heads == 1:
        model = models.UNet2d_with_heads(in_channels = 1,
                                         num_labels = args.num_labels,
                                         squeeze = False)
    elif args.model_has_heads == 0:
        model = models.UNet2d(in_channels = 1,
                              num_labels = args.num_labels,
                              squeeze = False)
    model = model.to(device)

    # ===================================
    # load model weights
    # ===================================
    modelpath = utils_generic.get_best_modelpath(models_path, 'model_best_dice')
    logging.info('loading model weights from: ')
    logging.info(modelpath)
    model.load_state_dict(torch.load(modelpath)['state_dict'])

    # ===================================
    # Set model to eval mode
    # ===================================
    model.eval()

    # ===================================
    # evaluate each subject one by one
    # ===================================
    logging.info('number of test subjects: ' + str(len(subject_names_ts)))
    logging.info('depth dimensions of these test subjects: ' + str(depths_ts))
    subject_dices = []
    for sub in range(len(subject_names_ts)):
        subject_name = subject_names_ts[sub]
        logging.info(subject_name)
        sub_start = int(np.sum(depths_ts[:sub]))
        sub_end = int(np.sum(depths_ts[:sub+1]))
        subject_image = images_ts[:,:,sub_start:sub_end]
        subject_label = labels_ts[:,:,sub_start:sub_end]

        for b in range(subject_image.shape[-1] // args.batch_size_test + 1):

            if (b+1) * args.batch_size_test < subject_image.shape[-1]:
                x_batch = subject_image[:, :, b*args.batch_size_test : (b+1) * args.batch_size_test]
            elif b * args.batch_size_test == subject_image.shape[-1]:
                break
            else:
                x_batch = subject_image[:, :, b*args.batch_size_test : ]
                        
            # swap axes to bring batch dimension from the back to the front
            x_batch = np.swapaxes(np.swapaxes(x_batch, 2, 1), 1, 0)
            
            # add channel axis
            x_batch = np.expand_dims(x_batch, axis = 1)
            
            # send to gpu
            x_batch_gpu = utils_data.make_torch_tensor_and_send_to_device(x_batch, device)
            
            # make prediction
            outputs = model(x_batch_gpu)
            predicted_probs_gpu_this_batch = torch.nn.Softmax(dim=1)(outputs[-1])
            
            # accumulate predictions
            if b == 0:
                predicted_probs_gpu = predicted_probs_gpu_this_batch
            else:
                predicted_probs_gpu = torch.cat((predicted_probs_gpu, predicted_probs_gpu_this_batch), dim = 0)

        # move prediction to cpu
        predicted_probs_cpu = predicted_probs_gpu.detach().cpu().numpy()
        
        # working with binary segmentations for now
        soft_prediction = predicted_probs_cpu[:, 1, :, :]
        hard_prediction = (soft_prediction > 0.5).astype(np.float32)
        subject_label = np.swapaxes(np.swapaxes(subject_label, 2, 1), 1, 0)

        # compute metrics
        scores = compute_metrics(pred = hard_prediction, label = subject_label)
        subject_dices.append(np.round(scores[0], 3))

    write_results_to_file(results_path,
                          args.test_sub_dataset,
                          subject_dices,
                          subject_names_ts)