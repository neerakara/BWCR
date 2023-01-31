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

# ======================================================
# Function used to evaluate entire training / validation sets during training
# ======================================================
def evaluate(args,
             model,
             device,
             subdataset,
             ttv):

    # set model to evaluate mode
    model.eval()

    # evaluate dice for all subjects in ttv split of this subdataset
    dice_scores = utils_data.evaluate(args, subdataset, ttv, model, device)

    # set model back to training mode
    model.train()

    return dice_scores

# ==========================================
# ==========================================
def invert_features(tensor,
                    transformation,
                    interp,
                    need_mask,
                    default_shape,
                    device):
    
    tensor_inv = utils_data.invert_geometric_transforms(tensor, transformation, interp)
    
    if need_mask:
        # some zero padding might have been introduced in the process of doing and undoing the geometric transformation
        mask_gpu = utils_data.apply_geometric_transforms_mask(torch.ones(default_shape).to(device, dtype = torch.float), transformation)
        mask_resized = utils_data.rescale_tensor(mask_gpu, tensor_inv.shape[-1])
        mask_gpu_inv = utils_data.invert_geometric_transforms(mask_resized, transformation, interp = 'nearest')
        mask = mask_gpu_inv.repeat(1, tensor_inv.shape[1], 1, 1)
        return tensor_inv, mask
    
    else:
        return tensor_inv

# ==========================================
# ==========================================    
def torch_and_send(xcpu, ycpu, dev):
    xgpu = utils_data.torch_and_send_to_device(xcpu, dev)
    ygpu = utils_data.torch_and_send_to_device(ycpu, dev)
    return xgpu, ygpu

# ==========================================
# ==========================================    
def make_1hot(y, n):
    return utils_data.make_label_onehot(y, n)

# ==========================================
# ==========================================    
def get_losses(logits, targets, mask = None):
    
    loss_pc = - targets * F.log_softmax(logits, dim=1) # pixel wise cross entropy
    
    if mask != None:
        loss_pc = torch.mul(loss_pc, mask)
    
    loss_p = torch.mean(loss_pc, dim = 1) # average both classes
    loss = torch.mean(loss_p) # mean over all pixels and all images in the batch
    
    return loss_pc, loss_p, loss
    
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
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_iterations', default=11, type=int)
    parser.add_argument('--log_frequency', default=50, type=int)
    parser.add_argument('--eval_frequency_tr', default=1000, type=int)
    parser.add_argument('--eval_frequency_vl', default=10, type=int)
    parser.add_argument('--save_frequency', default=10000, type=int)

    # no tricks: (100), data aug (010), data aug + consistency (011 / 012)
    parser.add_argument('--l0', default=1, type=float) # 0 / 1
    parser.add_argument('--l1', default=0, type=float) # 0 / 1
    parser.add_argument('--l2', default=0, type=float) # 0 / 1 
    parser.add_argument('--temp', default=1, type=float) # 1 / 2
        
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
    logging_dir = utils_generic.make_expdir_2(args)
    logging.info('EXPERIMENT NAME: ' + logging_dir)

    # ===================================
    # Create a summary writer, # Create dirs for summaries saving models, saving vis
    # ===================================
    summary_path = logging_dir + 'summary/'
    models_path = logging_dir + 'models/'
    vis_path = logging_dir + 'vis/'
    if not os.path.exists(summary_path): os.makedirs(summary_path)
    if not os.path.exists(models_path): os.makedirs(models_path)
    if not os.path.exists(vis_path): os.makedirs(vis_path)
    writer = SummaryWriter(summary_path)
    
    # ===================================
    # define model
    # ===================================
    model = models.UNet2d(in_channels = 1, num_labels = args.num_labels, squeeze = False)
    model = model.to(device)

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

    # ===================================
    # run training iterations
    # ===================================
    logging.info('Starting training iterations')
    for iteration in range(args.max_iterations):

        optimizer.zero_grad() # reset gradients

        # =======================
        # A (1) get batch of orig images and labels, (2) send to gpu, (3) get predictions
        # =======================
        inputs0_cpu, labels0_cpu = utils_data.get_batch(images_tr, labels_tr, args.batch_size) # get batch of original images and labels
        inputs0_gpu, labels0_gpu = torch_and_send(inputs0_cpu, labels0_cpu, device)
        outputs0 = model(inputs0_gpu) # compute predictions
        
        # =======================
        # B (1) transform original images and labels (once), (2) get predictions, (3) invert geometric transforms
        # =======================
        inputs1_gpu, labels1_gpu, t1 = utils_data.transform_batch(inputs0_gpu, labels0_gpu, args.data_aug_prob, device)
        outputs1 = model(inputs1_gpu)
        logits1_inv, mask1 = invert_features(outputs1[-1], t1, 'bilinear', True, inputs1_gpu.shape, device)
        labels1_inv = invert_features(labels1_gpu, t1, 'nearest', False, inputs1_gpu.shape, device)

        # =======================
        # C (1) transform original images and labels (again), (2) get predictions, (3) invert geometric transforms
        # =======================
        inputs2_gpu, labels2_gpu, t2 = utils_data.transform_batch(inputs0_gpu, labels0_gpu, args.data_aug_prob, device)
        outputs2 = model(inputs2_gpu)
        logits2_inv, mask2 = invert_features(outputs2[-1], t2, 'bilinear', True, inputs2_gpu.shape, device)
        labels2_inv = invert_features(labels2_gpu, t2, 'nearest', False, inputs2_gpu.shape, device)

        # =======================
        # D supervised losses on (1) original data, (2) transformed data 1 (in original coordinates), (3) transformed data 2
        # =======================
        sup_loss0_pc, sup_loss0_p, sup_loss0 = get_losses(logits = outputs0[-1], targets = make_1hot(labels0_gpu, args.num_labels))
        sup_loss1_pc, sup_loss1_p, sup_loss1 = get_losses(logits = logits1_inv, targets = make_1hot(labels1_inv, args.num_labels), mask = mask1)
        sup_loss2_pc, sup_loss2_p, sup_loss2 = get_losses(logits = logits2_inv, targets = make_1hot(labels2_inv, args.num_labels), mask = mask2)

        # =======================
        # E consistency losses / soft label losses on (1) transformed data 1, (2) transformed data 2 (using the other's preds as soft targets)
        # =======================
        con_loss1_pc, con_loss1_p, con_loss1 = get_losses(logits = logits1_inv, targets = F.softmax(logits2_inv / args.temp, dim = 1), mask = torch.mul(mask1, mask2))
        con_loss2_pc, con_loss2_p, con_loss2 = get_losses(logits = logits2_inv, targets = F.softmax(logits1_inv / args.temp, dim = 1), mask = torch.mul(mask1, mask2))

        # =======================
        # add losses to tensorboard
        # =======================
        writer.add_scalar("Tr/sup_loss0", sup_loss0, iteration+1)
        writer.add_scalar("Tr/sup_loss1", sup_loss1, iteration+1)
        writer.add_scalar("Tr/sup_loss2", sup_loss2, iteration+1)
        writer.add_scalar("Tr/con_loss1", con_loss1, iteration+1)
        writer.add_scalar("Tr/con_loss2", con_loss2, iteration+1)

        # =======================
        # set desired combinations of losses and compute gradients
        # =======================
        total_loss = (args.l0 * sup_loss0 + args.l1 * sup_loss1 + args.l2 * con_loss1) / (args.l0 + args.l1 + args.l2)
        total_loss.backward()

        # =======================
        # optimizer
        # =======================
        optimizer.step()
        scheduler.step()
        writer.add_scalar("Tr/lr", scheduler.get_last_lr()[0], iteration+1)

        # ==========
        # log losses
        # ==========
        logging.info('iter ' + str(iteration + 1) + 
                        ', sup = ' + str(np.round(sup_loss0.detach().cpu().numpy(), 2)) +
                        ', dataaug (1) = ' + str(np.round(sup_loss1.detach().cpu().numpy(), 2)) +
                        ', dataaug (2) = ' + str(np.round(sup_loss2.detach().cpu().numpy(), 2)) +
                        ', consistency (1) = ' + str(np.round(con_loss1.detach().cpu().numpy(), 2)) + 
                        ', consistency (2) = ' + str(np.round(con_loss2.detach().cpu().numpy(), 2)))
        
        # ===================================
        # evaluate on entire validation and test sets every once in a while
        # ===================================
        if (iteration % args.eval_frequency_vl == 0):
            dice_score_ts = evaluate(args, model, device, args.sub_dataset, 'test')
            dice_score_tr = evaluate(args, model, device, args.sub_dataset, 'train')
            dice_score_vl = evaluate(args, model, device, args.sub_dataset, 'validation')
            writer.add_scalar("Tr/DiceTrain", np.mean(dice_score_tr), iteration+1)
            writer.add_scalar("Tr/Dice_Test", np.mean(dice_score_ts), iteration+1)
            writer.add_scalar("Tr/Dice_Val", np.mean(dice_score_vl), iteration+1)

        # tensorboard
        writer.flush()

    # visualize
    utils_vis.save_all([inputs0_gpu[:,0,:,:], # orig images 
                        labels0_gpu[:,0,:,:], # orig labels
                        F.softmax(outputs0[-1] / 1.0, dim = 1)[:,1,:,:], # probs (orig images)
                        inputs1_gpu[:,0,:,:], # transformed images
                        F.softmax(logits1_inv / 1.0, dim = 1)[:,1,:,:], # probs (transformed images)
                        F.softmax(logits1_inv / args.temp, dim = 1)[:,1,:,:], # 'calibrated' probs (transformed images)
                        inputs2_gpu[:,0,:,:], # transformed images
                        F.softmax(logits2_inv / 1.0, dim = 1)[:,1,:,:], # probs (transformed images)
                        F.softmax(logits2_inv / args.temp, dim = 1)[:,1,:,:], # 'calibrated' probs (transformed images)
                        sup_loss0_p,
                        sup_loss1_p,
                        con_loss1_p], # total loss including supervised and smoothened loss
                        vis_path + 'iter500.png')

