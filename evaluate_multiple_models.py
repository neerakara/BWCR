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

# ======================================================
# ======================================================
def get_batch_subject(image, b, bs, device):

    if (b+1) * bs < image.shape[-1]:
        x_batch = image[:, :, b * bs : (b+1) * bs]
    else:
        x_batch = image[:, :, b * bs : ]
    x_batch = np.expand_dims(np.swapaxes(np.swapaxes(x_batch, 2, 1), 1, 0), axis = 1)
    return utils_data.torch_and_send_to_device(x_batch, device)

# ======================================================
# ======================================================
def predict_segmentation(image, model, device, px, py, nx, ny):

    bsize = 8
    n_batches = np.ceil(image.shape[-1] / bsize).astype(int)
    
    for b in range(n_batches):
        x_batch = get_batch_subject(image, b, bsize, device)
        preds_gpu_this_batch = torch.nn.Softmax(dim=1)(model(x_batch)[-1])
        if b == 0:
            preds_gpu = preds_gpu_this_batch
        else:
            preds_gpu = torch.cat((preds_gpu, preds_gpu_this_batch), dim = 0)

    preds_soft = preds_gpu.detach().cpu().numpy()[:, 1, :, :]
    preds_hard = (preds_soft > 0.5).astype(np.float32)
    preds_hard = utils_data.rescale_and_crop(preds_hard, scale = (0.625 / px, 0.625 / py), size = (nx, ny), order = 0).astype(np.uint8)
    preds_hard = np.swapaxes(np.swapaxes(preds_hard, 0, 1), 1, 2)

    return preds_hard

# ======================================================
# ======================================================
def get_gt_label(subdataset, subname):
    label = data_loader.load_without_preproc(subdataset, subname)[1]
    label[label!=0] = 1
    if subdataset in ['UCL', 'HK', 'BIDMC']:
        label = np.swapaxes(np.swapaxes(label, 0, 1), 1, 2)
    return label

# ======================================================
# Function used to evaluate entire training / validation sets during training
# ======================================================
def evaluate(args, subdataset, ttv, model, device):

    data = data_loader.load_data(args, subdataset, ttv)
    
    dice_scores = []
    
    with torch.no_grad():
    
        for sub in range(data['depths'].shape[0]):
    
            # get image of one subject
            image = data['images'][:,:,int(np.sum(data['depths'][:sub])):int(np.sum(data['depths'][:sub+1]))]

            # predict segmentation
            pred = predict_segmentation(image, model, device, data["px"][sub], data["py"][sub], data["nx"][sub], data["ny"][sub])

            # read original label (without preprocessing)
            label = get_gt_label(subdataset, data["subject_names"][sub].decode('utf-8'))
            
            # compute dice
            dice_scores.append(utils_generic.dice(im1 = pred, im2 = label))

    return np.array(dice_scores)

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
    parser.add_argument('--save_path', default='/data/scratch/nkarani/projects/crael/seg/logdir/v6/')
    parser.add_argument('--data_aug_prob', default=0.5, type=float)
    parser.add_argument('--optimizer', default='adam') # adam / sgd
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_schedule', default=2, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_test', default=4, type=int)
    parser.add_argument('--model_has_heads', default=0, type=int)    
    parser.add_argument('--method_invariance', default=100, type=int) # 0: no reg, 1: data aug, 2: consistency, 3: consistency in each layer
    parser.add_argument('--lambda_dataaug', default=1.0, type=float) # weight for data augmentation loss
    parser.add_argument('--consis_loss', default=2, type=int) # 1: MSE | 2: MSE of normalized images (BYOL)
    parser.add_argument('--lambda_consis', default=0.1, type=float) # weight for regularization loss (consistency overall)
    parser.add_argument('--alpha_layer', default=100.0, type=float) # growth of regularization loss weight with network depth
    parser.add_argument('--run_number', default=1, type=int)
    parser.add_argument('--debugging', default=0, type=int)    
    parser.add_argument('--model_prefix', default='model') # best_val / model
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
    logging.info('Evaluation experiment: ' + logging_dir)

    # ===================================
    # define model
    # ===================================
    logging.info('Defining segmentation model')
    model = models.UNet2d(in_channels = 1, num_labels = args.num_labels, squeeze = False)
    model = model.to(device)

    # ===================================
    # Set model to eval mode
    # ===================================
    model.eval()

    # initialize lists to store results of different datasets
    ind_val = []
    ind_test = []
    ood_test = []

    # ===================================
    # load model weights
    # ===================================
    maxiter = 100001
    num_models_to_eval = 100
    measure_var = 'full' # 'end' / 'full'
    if measure_var == 'end':
        iter_range = range(maxiter - num_models_to_eval, maxiter)
    elif measure_var == 'full':
        iter_range = range(10000, 110000, 10000)

    for iteration in iter_range:

        modelpath = models_path + 'model_iter' + str(iteration) + '.pt'
        logging.info('loading model weights from: ' + modelpath)
        model.load_state_dict(torch.load(modelpath)['state_dict'])

        # ===================================
        # evaluate each dataset one by one
        # ===================================
        ind_val.append(evaluate(args, 'RUNMC', 'validation', model, device))
        ind_test.append(evaluate(args, 'RUNMC', 'test', model, device))
        ood_test_tmp = evaluate(args, 'BMC', 'test', model, device)
        ood_test_tmp = np.concatenate((ood_test_tmp, evaluate(args, 'UCL', 'test', model, device)))
        ood_test_tmp = np.concatenate((ood_test_tmp, evaluate(args, 'HK', 'test', model, device)))
        ood_test_tmp = np.concatenate((ood_test_tmp, evaluate(args, 'BIDMC', 'test', model, device)))
        ood_test.append(ood_test_tmp)

    np.save(results_path + 'ind_val_' + measure_var + '.npy', np.array(ind_val))
    np.save(results_path + 'ind_test_' + measure_var + '.npy', np.array(ind_test))
    np.save(results_path + 'ood_test_' + measure_var + '.npy', np.array(ood_test))
