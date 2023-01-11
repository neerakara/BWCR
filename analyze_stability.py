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

# ==========================================
# ==========================================
def transform_image(image, transform_num):
    return image

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
    
    parser.add_argument('--dataset', default='ms') # placenta / prostate
    parser.add_argument('--sub_dataset', default='RUNMC') # prostate: BIDMC / BMC / HK / I2CVB / RUNMC / UCL
    parser.add_argument('--test_sub_dataset', default='InD') # prostate: BIDMC / BMC / HK / I2CVB / RUNMC / UCL
    parser.add_argument('--cv_fold_num', default=2, type=int)
    parser.add_argument('--num_labels', default=2, type=int)

    parser.add_argument('--save_path', default='/data/scratch/nkarani/projects/crael/seg/logdir/v3/')
    
    parser.add_argument('--data_aug_prob', default=0.5, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)
    
    parser.add_argument('--model_has_heads', default=0, type=int)    
    parser.add_argument('--method_invariance', default=0, type=int) # 0: no reg, 1: data aug, 2: consistency, 3: consistency in each layer
    parser.add_argument('--lambda_dataaug', default=1.0, type=float) # weight for data augmentation loss
    parser.add_argument('--consis_loss', default=2, type=int) # 1: MSE | 2: MSE of normalized images (BYOL)
    parser.add_argument('--lambda_consis', default=0.001, type=float) # weight for regularization loss (consistency overall)
    parser.add_argument('--alpha_layer', default=10.0, type=float) # growth of regularization loss weight with network depth
    
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
    logging.info('Evaluation experiment: ' + logging_dir)
    logging.info('Reading test data')
    data_test = data_loader.load_data(args, args.test_sub_dataset, 'test')
    images_ts = data_test["images"]
    labels_ts = data_test["labels"]
    depths_ts = data_test["depths"]
    subject_names_ts = data_test["subject_names"]

    # ===================================
    # define model
    # ===================================
    logging.info('Defining segmentation model')
    if args.model_has_heads == 1:
        model = models.UNet2d_with_heads(in_channels = 1,
                                         num_labels = args.num_labels,
                                         squeeze = False,
                                         returnlist = 2)
    elif args.model_has_heads == 0:
        model = models.UNet2d(in_channels = 1,
                              num_labels = args.num_labels,
                              squeeze = False,
                              returnlist = 2)
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
    num_subjects_ts = subject_names_ts.shape[0]
    logging.info('number of test subjects: ' + str(num_subjects_ts))
    logging.info('depth dimensions of these test subjects: ' + str(depths_ts))
    subject_dices = []
    stddevs = []
    
    for sub in range(5):#(num_subjects_ts):
    
        subject_name = subject_names_ts[sub]
        subject_image = images_ts[:, :, int(np.sum(depths_ts[:sub])) : int(np.sum(depths_ts[:sub+1]))]
        subject_label = labels_ts[:, :, int(np.sum(depths_ts[:sub])) : int(np.sum(depths_ts[:sub+1]))]

        idx = np.argmax(np.sum(subject_label, axis=(0,1)))

        y_batch = subject_label[:, :, idx : idx + args.batch_size_test]
        y_batch = np.expand_dims(np.swapaxes(np.swapaxes(y_batch, 2, 1), 1, 0), axis = 1)
        
        x_batch = subject_image[:, :, idx : idx + args.batch_size_test]
        x_batch = np.expand_dims(np.swapaxes(np.swapaxes(x_batch, 2, 1), 1, 0), axis = 1)
        x_batch_gpu = utils_data.torch_and_send_to_device(x_batch, device)

        num_layers = len(model(x_batch_gpu))
        features = {}
        for l in range(num_layers+2): # including input and output
            features['features_layer' + str(l)] = []

        # =================
        # transform the image multiple times and compute predictions 
        # =================
        num_transforms = 10
        for transform_num in range(num_transforms): 
            
            x_batch = subject_image[:, :, subject_image.shape[-1] // 2 : subject_image.shape[-1] // 2 + args.batch_size_test]
            x_batch = np.expand_dims(np.swapaxes(np.swapaxes(x_batch, 2, 1), 1, 0), axis = 1)
            x_batch_gpu = utils_data.torch_and_send_to_device(x_batch, device)
            x_batch_gpu_transformed = utils_data.apply_intensity_transform_fixed(x_batch_gpu, device, transform_num, num_transforms)
            
            features['features_layer0'].append(x_batch_gpu_transformed.detach().cpu().numpy()[:, 0, :, :])
            
            model_outputs = model(x_batch_gpu_transformed)
            
            for l in range(num_layers):
                features['features_layer' + str(l+1)].append(model_outputs[l].detach().cpu().numpy())
            
            features['features_layer' + str(num_layers+1)].append(torch.nn.Softmax(dim=1)(model_outputs[-1]).detach().cpu().numpy()[:, 1, :, :])

        # =================
        # compute statistics over transformations
        # =================
        means = {}
        vars = {}
        for l in range(num_layers+2):
            means['layer' + str(l)] = np.squeeze(np.mean(np.array(features['features_layer' + str(l)]), axis=0))
            vars['layer' + str(l)] = np.squeeze(np.var(np.array(features['features_layer' + str(l)]), axis=0))
            if l not in [0, num_layers+1]:
                means['layer' + str(l)] = np.mean(means['layer' + str(l)], axis=0) # average across channels
                vars['layer' + str(l)] = np.mean(vars['layer' + str(l)], axis=0) # average across channels

        # =================
        # visualize variances
        # =================
        utils_vis.show_prediction_variation_2(means,
                                              vars,
                                              results_path + 'stability_' + subject_name.decode('utf-8'))