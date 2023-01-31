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
from torch.utils.tensorboard import SummaryWriter
from monai.losses.dice import DiceLoss
import torch.nn.functional as F
# self-defined stuff
import models 
import utils_data
import utils_vis
import utils_generic
import data_loader

# ==========================================
# ==========================================
def get_probs_and_outputs(model, inputs_gpu):
    model_outputs = model(inputs_gpu)
    outputs_probs = torch.nn.Softmax(dim=1)(model_outputs[-1])
    return model_outputs, outputs_probs

# ==========================================
# ==========================================
def get_inverted_tensor_and_inversion_mask(tensor,
                                           transformation,
                                           default_shape,
                                           device):
    
    tensor_inv = utils_data.invert_geometric_transforms(tensor, transformation)
    
    # some zero padding might have been introduced in the process of doing and undoing the geometric transformation
    mask1_gpu = utils_data.apply_geometric_transforms_mask(torch.ones(default_shape).to(device, dtype = torch.float), transformation)
    mask1_resized = utils_data.rescale_tensor(mask1_gpu, tensor_inv.shape[-1])
    mask1_gpu_inv = utils_data.invert_geometric_transforms_mask(mask1_resized, transformation)
    mask = mask1_gpu_inv.repeat(1, tensor_inv.shape[1], 1, 1)
    
    return tensor_inv, mask
    
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
    
    parser.add_argument('--dataset', default='prostate') # placenta / prostate / ms
    parser.add_argument('--sub_dataset', default='RUNMC') # prostate: BIDMC / BMC / HK / I2CVB / RUNMC / UCL / InD / OoD
    parser.add_argument('--cv_fold_num', default=3, type=int)
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--save_path', default='/data/scratch/nkarani/projects/crael/seg/logdir/v7/')
    
    parser.add_argument('--data_aug_prob', default=0.5, type=float)
    parser.add_argument('--optimizer', default='adam') # adam / sgd
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_schedule', default=2, type=int)
    parser.add_argument('--lr_schedule_step', default=15000, type=int)
    parser.add_argument('--lr_schedule_gamma', default=0.1, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_iterations', default=501, type=int)
    parser.add_argument('--log_frequency', default=50, type=int)
    parser.add_argument('--eval_frequency_tr', default=1000, type=int)
    parser.add_argument('--eval_frequency_vl', default=100, type=int)
    parser.add_argument('--save_frequency', default=10000, type=int)
    
    parser.add_argument('--model_has_heads', default=0, type=int)    
    parser.add_argument('--method_invariance', default=1, type=int) # 0: no reg, 1: data aug, 2: consistency in each layer (geom + int), 3: consistency in each layer (int)
    parser.add_argument('--lambda_dataaug', default=1.0, type=float) # weight for data augmentation loss
    parser.add_argument('--consis_loss', default=1, type=int) # 1: MSE | 2: MSE of normalized images (BYOL)
    parser.add_argument('--lambda_consis', default=1.0, type=float) # weight for regularization loss (consistency overall)
    parser.add_argument('--alpha_layer', default=1.0, type=float) # growth of regularization loss weight with network depth
    parser.add_argument('--load_model_num', default=0, type=int) # load_model_num = 0 --> train from scratch
    
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
    # Create a summary writer
    # ===================================
    logging.info('Creating a summary writer for tensorboard')
    summary_path = logging_dir + 'summary/'
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    writer = SummaryWriter(summary_path)

    # ===================================
    # Create a dir for saving models
    # ===================================
    models_path = logging_dir + 'models/'
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # ===================================
    # Create a dir for saving vis
    # ===================================
    vis_path = logging_dir + 'vis/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    # ===================================
    # define model
    # ===================================
    model = models.UNet2d(in_channels = 1, num_labels = args.num_labels, squeeze = False)
    model = model.to(device)

    # ===================================
    # define losses
    # ===================================
    logging.info('Defining losses')
    supervised_loss_function = DiceLoss(include_background=False).to(device) 
    # logsoftmax = nn.LogSoftmax(dim=1).to(device) 
    if args.method_invariance in [2, 3, 20, 30, 200, 300]:
        cons_loss_function = torch.nn.MSELoss(reduction='sum') # sum of squared errors to remove magnitude dependence on image size
        cons_loss_function = cons_loss_function.to(device) 

    # ===================================
    # define optimizer
    # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    # ===================================
    logging.info('Defining optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    lambda1 = lambda iteration: 1 - ((args.lr - 1e-7) * iteration / (args.max_iterations * args.lr))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # ===================================
    # set model to train mode
    # ===================================
    model.train()

    # ===================================
    # load images and labels
    # ===================================
    logging.info('Reading training and validation data')
    data_tr = data_loader.load_data(args, args.sub_dataset, 'train')
    images_tr = data_tr["images"]
    labels_tr = data_tr["labels"]
    subject_names_tr = data_tr["subject_names"]

    # # ===================================
    # # choose one batch of images and labels | visualize it
    # # ===================================
    # inputs_cpu, labels_cpu = utils_data.get_batch(images_tr, labels_tr, args.batch_size)
    # utils_vis.save_images_labels_all(inputs_cpu, labels_cpu, vis_path + 'train_data.png')
    # logging.info(inputs_cpu.shape)
    # logging.info(labels_cpu.shape)

    # ===================================
    # run training iterations
    # ===================================
    logging.info('Starting training iterations')
    for iteration in range(500): #(args.max_iterations):

        optimizer.zero_grad() # reset gradients

        # ==========================================
        # supervised loss on original images
        # ==========================================
        inputs_cpu, labels_cpu = utils_data.get_batch(images_tr, labels_tr, args.batch_size) # get batch of original images and labels
        inputs_gpu = utils_data.torch_and_send_to_device(inputs_cpu, device)
        labels_gpu = utils_data.torch_and_send_to_device(labels_cpu, device)
        labels_gpu_1hot = utils_data.make_label_onehot(labels_gpu, args.num_labels)
        model_outputs, outputs_probs = get_probs_and_outputs(model, inputs_gpu) # pass through model and compute predictions
        supervised_loss_pixelwise_classwise = - labels_gpu_1hot * F.log_softmax(model_outputs[-1], dim=1) # pixel wise cross entropy
        supervised_loss_pixelwise = torch.mean(supervised_loss_pixelwise_classwise, dim = 1) # average both classes
        supervised_loss = torch.mean(supervised_loss_pixelwise) # mean over all pixels and all images in the batch

        # ==========================================
        # data augmentation and correponding 'supervised' loss
        # ==========================================
        inputs1_gpu, labels1_gpu, t1 = utils_data.transform_batch(inputs_gpu, labels_gpu, data_aug_prob = args.data_aug_prob, device = device)
        labels1_gpu_1hot = utils_data.make_label_onehot(labels1_gpu, args.num_labels)
        model_outputs1, outputs_probs1 = get_probs_and_outputs(model, inputs1_gpu)
        dataaug_loss_pixelwise_classwise = - labels1_gpu_1hot * F.log_softmax(model_outputs1[-1], dim=1)
        dataaug_loss_pixelwise = torch.mean(dataaug_loss_pixelwise_classwise, dim = 1) # average both classes
        dataaug_loss = torch.mean(dataaug_loss_pixelwise)

        # to compare pixelwise differences between supervised and unsupervised losses, we need to get all loss maps in the same coordinate system
        dataaug_loss_pixelwise_classwise_inv, _ = get_inverted_tensor_and_inversion_mask(dataaug_loss_pixelwise_classwise, t1, inputs1_gpu.shape, device)
        dataaug_loss_pixelwise_inv = torch.mean(dataaug_loss_pixelwise_classwise_inv, dim = 1) # average both classes

        # ==========================================
        # 'consistency loss' / 'loss with smoothed labels'
        # ==========================================
        features1_inv, mask = get_inverted_tensor_and_inversion_mask(model_outputs1[-1], t1, inputs1_gpu.shape, device) # target logits
        temperature = 2.0
        target_probabilities = F.softmax(features1_inv / temperature) # add temperature if needed
        consistency_loss_pixelwise_classwise = torch.mul(- target_probabilities * F.log_softmax(model_outputs[-1], dim=1), mask) # pixel wise cross entropy  
        consistency_loss_pixelwise = torch.mean(consistency_loss_pixelwise_classwise, dim = 1) # average both classes
        consistency_loss = torch.mean(consistency_loss_pixelwise)

        # compute gradients
        logging.info('iter ' + str(iteration + 1) + 
                        ', sup = ' + str(np.round(supervised_loss.detach().cpu().numpy(), 2)) +
                        ', dataaug = ' + str(np.round(dataaug_loss.detach().cpu().numpy(), 2)) +
                        ', consistency = ' + str(np.round(consistency_loss.detach().cpu().numpy(), 2)))
        
        total_loss = supervised_loss + dataaug_loss + consistency_loss
        total_loss.backward()

        # optimizer
        optimizer.step()
        scheduler.step()
        writer.add_scalar("Tr/lr", scheduler.get_last_lr()[0], iteration+1)

        # tensorboard
        writer.flush()

    utils_vis.save_all([inputs_gpu[:,0,:,:],
                        labels_gpu[:,0,:,:],
                        model_outputs[-1][:,-1,:,:], # logits
                        outputs_probs[:,-1,:,:], # fg probabilities
                        supervised_loss_pixelwise, # pixel wise supervised loss (orig images)
                        inputs1_gpu[:,0,:,:],
                        labels1_gpu[:,0,:,:],
                        model_outputs1[-1][:,-1,:,:], # logits
                        outputs_probs1[:,-1,:,:], # fg probabilities
                        dataaug_loss_pixelwise_inv, # pixel wise supervised loss (transformed images)
                        consistency_loss_pixelwise, # pixel wise loss when targets are predictions of transformed images
                        supervised_loss_pixelwise + dataaug_loss_pixelwise_inv, # pixel losses if label smoothened would not be used
                        supervised_loss_pixelwise + dataaug_loss_pixelwise_inv + consistency_loss_pixelwise], # total loss including supervised and smoothened loss
                        vis_path + 'iter500.png')

    # ===================================
    # make two batches of 8 images each | visualize
    # ===================================
    

