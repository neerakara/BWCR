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
# monai makes life easier
from monai.losses.dice import DiceLoss
from torch.utils.tensorboard import SummaryWriter
# self-defined stuff
import models 
import utils_data
import utils_vis
import utils_generic
import data_loader
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.use('agg')
import matplotlib.pyplot as plt 

# ==========================================================
# ==========================================================
def save_all_images_and_labels(images,
                               labels,
                               savefilename,
                               normalize = False,
                               rot90_k = -1):
    
    tmp_images = images.detach().cpu().numpy()
    tmp_labels = labels.detach().cpu().numpy()
    
    nc = tmp_images.shape[0]
    nr = 2
    plt.figure(figsize=(6*nc, 6*nr))
    
    for b in range(nc):
        
        if normalize == True:
            im = utils_vis.normalize_img_for_vis(tmp_images[b,0,:,:])
            lb = utils_vis.normalize_img_for_vis(tmp_labels[b,0,:,:])
        
        else:
            im = tmp_images[b,0,:,:]
            lb = tmp_labels[b,0,:,:]
        
        if rot90_k != 0:
            im = np.rot90(im, k=rot90_k)
            lb = np.rot90(lb, k=rot90_k)
        
        plt.subplot(nr, nc, b+1, xticks=[], yticks=[]); plt.imshow(im, cmap = 'gray'); plt.colorbar()
        plt.subplot(nr, nc, nc+b+1, xticks=[], yticks=[]); plt.imshow(lb, cmap = 'gray'); plt.colorbar()
    
    plt.savefig(savefilename, bbox_inches='tight', dpi=50)
    plt.close()
    
    return 0

# ==========================================
# ==========================================
def save_heads_old(x, h, s, invert=False, t=0):

    if invert:
        x = utils_data.invert_geometric_transforms(x, t)
        for l in range(len(h)):
            h[l] = utils_data.invert_geometric_transforms(h[l], t)

    im = x.detach().cpu().numpy()
    y = torch.nn.Softmax(dim=1)(h[-1])
    prob = y.detach().cpu().numpy()
    
    nc = 6
    nr = 2
    plt.figure(figsize=(6*nc, 6*nr))
    plt.subplot(nr, nc, 1, xticks=[], yticks=[]); plt.imshow(im[0,0,:,:], cmap = 'gray'); plt.colorbar()
    for l in range(len(h)):
        head = h[l].detach().cpu().numpy()
        plt.subplot(nr, nc, l+2, xticks=[], yticks=[]); plt.imshow(head[0,-1,:,:], cmap = 'gray'); plt.colorbar()
    plt.subplot(nr, nc, nc*nr, xticks=[], yticks=[]); plt.imshow(prob[0,-1,:,:], cmap = 'gray'); plt.colorbar()
    plt.savefig(s, bbox_inches='tight', dpi=50)
    plt.close()
    
    return 0

# ==========================================
# ==========================================
def save_heads(x, h, x_t1, h_t1, s, t):
    
    nc = 12
    nr = 4
    plt.figure(figsize=(4*nc, 4*nr))

    # orig
    im = x.detach().cpu().numpy()
    plt.subplot(nr, nc, 1, xticks=[], yticks=[]); plt.imshow(im[0,0,:,:], cmap = 'gray'); plt.colorbar()
    for l in range(len(h)):
        head = h[l].detach().cpu().numpy()
        plt.subplot(nr, nc, l+2, xticks=[], yticks=[]); plt.imshow(head[0,-1,:,:], cmap = 'gray'); plt.colorbar()
    prob = torch.nn.Softmax(dim=1)(h[-1]).detach().cpu().numpy()
    plt.subplot(nr, nc, nc, xticks=[], yticks=[]); plt.imshow(prob[0,-1,:,:], cmap = 'gray'); plt.colorbar()

    # transformed
    im = x_t1.detach().cpu().numpy()
    plt.subplot(nr, nc, nc+1, xticks=[], yticks=[]); plt.imshow(im[0,0,:,:], cmap = 'gray'); plt.colorbar()
    for l in range(len(h_t1)):
        head = h_t1[l].detach().cpu().numpy()
        plt.subplot(nr, nc, nc+l+2, xticks=[], yticks=[]); plt.imshow(head[0,-1,:,:], cmap = 'gray'); plt.colorbar()
    prob = torch.nn.Softmax(dim=1)(h_t1[-1]).detach().cpu().numpy()
    plt.subplot(nr, nc, 2*nc, xticks=[], yticks=[]); plt.imshow(prob[0,-1,:,:], cmap = 'gray'); plt.colorbar()

    # transform inverted
    x_t1 = utils_data.invert_geometric_transforms(x_t1, t)
    for l in range(len(h)):
        h_t1[l] = utils_data.invert_geometric_transforms(h_t1[l], t)

    im = x_t1.detach().cpu().numpy()
    plt.subplot(nr, nc, 2*nc+1, xticks=[], yticks=[]); plt.imshow(im[0,0,:,:], cmap = 'gray'); plt.colorbar()
    for l in range(len(h_t1)):
        head = h_t1[l].detach().cpu().numpy()
        plt.subplot(nr, nc, 2*nc+l+2, xticks=[], yticks=[]); plt.imshow(head[0,-1,:,:], cmap = 'gray'); plt.colorbar()
    prob = torch.nn.Softmax(dim=1)(h_t1[-1]).detach().cpu().numpy()
    plt.subplot(nr, nc, 3*nc, xticks=[], yticks=[]); plt.imshow(prob[0,-1,:,:], cmap = 'gray'); plt.colorbar()

    # difference
    im = x.detach().cpu().numpy() - x_t1.detach().cpu().numpy()
    plt.subplot(nr, nc, 3*nc+1, xticks=[], yticks=[]); plt.imshow(im[0,0,:,:], cmap = 'gray'); plt.colorbar()
    for l in range(len(h_t1)):
        head = h[l].detach().cpu().numpy() - h_t1[l].detach().cpu().numpy()
        plt.subplot(nr, nc, 3*nc+l+2, xticks=[], yticks=[]); plt.imshow(head[0,-1,:,:], cmap = 'gray'); plt.colorbar()
    prob = torch.nn.Softmax(dim=1)(h[-1]).detach().cpu().numpy() - torch.nn.Softmax(dim=1)(h_t1[-1]).detach().cpu().numpy()
    plt.subplot(nr, nc, 4*nc, xticks=[], yticks=[]); plt.imshow(prob[0,-1,:,:], cmap = 'gray'); plt.colorbar()
    
    plt.savefig(s, bbox_inches='tight', dpi=50)
    plt.close()
    
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
    parser.add_argument('--save_path', default='/data/scratch/nkarani/projects/crael/seg/logdir/v2/')
    
    parser.add_argument('--data_aug_prob', default=0.5, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_iterations', default=50001, type=int)
    parser.add_argument('--log_frequency', default=500, type=int)
    parser.add_argument('--eval_frequency', default=5000, type=int)
    parser.add_argument('--save_frequency', default=10000, type=int)
    
    parser.add_argument('--model_has_heads', default=1, type=int)    
    parser.add_argument('--method_invariance', default=2, type=int) # 0: no reg, 1: data aug, 2: consistency in each layer
    parser.add_argument('--lambda_consis', default=1.0, type=float) # weight for regularization loss (consistency overall)
    parser.add_argument('--alpha_layer', default=100.0, type=float) # growth of regularization loss weight with network depth
    
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
    logging.info("Using device: " + str(device))
    logging.info("Number of devices: " + str(torch.cuda.device_count()))

    # ===================================
    # ===================================
    logging_dir = utils_generic.make_expdir(args)
    logging.info('EXPERIMENT NAME: ' + logging_dir)

    # ===================================
    # Create a dir for saving vis
    # ===================================
    vis_path = logging_dir + 'vis/'
    models_path = logging_dir + 'models/'
    results_path = logging_dir + 'results/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # ===================================
    # load image and label
    # ===================================
    logging.info('Reading test data')
    data_test = data_loader.load_data(args.dataset, args.test_sub_dataset, args.cv_fold_num, 'test')
    images_ts = data_test["images"]
    labels_ts = data_test["labels"]
    depths_ts = data_test["depths"]
    subject_names_ts = data_test["subject_names"]

    # ===================================
    # define model
    # ===================================
    model = models.UNet2d_with_heads(in_channels = 1, num_labels = args.num_labels, squeeze = False)
    model = model.to(device)

    # ===================================
    # load model weights
    # ===================================
    modelpath = utils_generic.get_best_modelpath(models_path, 'model_best_dice')
    modelpath = models_path + 'model_best_dice_iter5000.pt'
    logging.info('loading model weights from: ')
    logging.info(modelpath)
    model.load_state_dict(torch.load(modelpath)['state_dict'])

    # ===================================
    # Set model to eval mode
    # ===================================
    model.eval()

    # ===================================
    # predict seg for an image slice
    # ===================================
    sub = 0
    subject_name = subject_names_ts[sub]
    logging.info(subject_name)
    sub_start = int(np.sum(depths_ts[:sub]))
    sub_end = int(np.sum(depths_ts[:sub+1]))
    subject_image = images_ts[:,:,sub_start:sub_end]
    subject_label = labels_ts[:,:,sub_start:sub_end]
    c = subject_image.shape[-1]//2
    x_batch = subject_image[:, :, c:c+1]
    y_batch = subject_label[:, :, c:c+1]
    # swap axes to bring batch dimension from the back to the front
    x_batch = np.swapaxes(np.swapaxes(x_batch, 2, 1), 1, 0)
    y_batch = np.swapaxes(np.swapaxes(y_batch, 2, 1), 1, 0)
    # add channel axis
    x_batch = np.expand_dims(x_batch, axis = 1)
    y_batch = np.expand_dims(y_batch, axis = 1)
    # send to gpu
    x_batch_gpu = utils_data.torch_and_send_to_device(x_batch, device)
    y_batch_gpu = utils_data.torch_and_send_to_device(y_batch, device)
    
    # make prediction for orig image
    out = model(x_batch_gpu)
    # make prediction for transformed image
    for tr in [[10,0,0,1.0,0,0], [0,10,10,1.0,0,0], [0,0,0,1.5,0,0], [0,0,0,1.0,10,10]]:

        for sample in range(10):
            xt1, yt1, geom_params1 = utils_data.transform_batch(x_batch_gpu, y_batch_gpu, args.data_aug_prob, device, t=tr)
            out_t1 = model(xt1)

            save_heads(x = x_batch_gpu,
                       h = out,
                       x_t1 = xt1,
                       h_t1 = out_t1,
                       s = results_path + 'heads_' + str(tr)[1:-1].replace(',','_') + '_intensity_sample' + str(sample+1) + '.png',
                       t = tr)
    
    # save_heads_old(x = x_batch_gpu, h = out, s = results_path + 'heads_orig.png')
    # save vis of predictions
    # save_heads_old(x = xt1, h = out_t1, s = results_path + 'heads_rot10.png')
    # invert predictions and visualize
    # save_heads_old(x = xt1, h = out_t1, s = results_path + 'heads_rot10_inverted.png', invert = True, t = tr)
    
    # for iteration in range(3):
    #     inputs1_gpu, labels1_gpu, t1 = utils_data.transform_batch(inputs_gpu, labels_gpu, data_aug_prob = args.data_aug_prob, device = device)
    #     save_all_images_and_labels(inputs1_gpu, labels1_gpu, vis_path + 't1_iter' +str(iteration) + '.png')
    #     inputs2_gpu, labels2_gpu, t2 = utils_data.transform_batch(inputs_gpu, labels_gpu, data_aug_prob = args.data_aug_prob, device = device)
    #     save_all_images_and_labels(inputs2_gpu, labels2_gpu, vis_path + 't2_iter' +str(iteration) + '.png')

    # save_all_images_and_labels(inputs_gpu, labels_gpu, vis_path + 'orig_after.png')