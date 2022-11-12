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

# import psutil
# import nvidia_smi
# nvidia_smi.nvmlInit()
# deviceCount = nvidia_smi.nvmlDeviceGetCount()

# ======================================================
# Function used to evaluate entire training / validation sets during training
# ======================================================
def evaluate(args,
             model,
             images_tr,
             labels_tr,
             images_vl,
             labels_vl,
             device,
             loss_function):

    # set model to evaluate mode
    model.eval()

    # initialize counters
    dice_score_tr = 0.0
    dice_score_vl = 0.0

    # loop through the train set and evaluate each batch
    with torch.no_grad():
        
        logging.info("Evaluating entire training dataset...")
        n_batches_tr = images_tr.shape[-1] // args.batch_size
        for iteration in range(n_batches_tr):
            inputs_cpu, labels_cpu = utils_data.get_batch(images_tr, labels_tr, args.batch_size, batch_type = 'sequential', start_idx = iteration * args.batch_size)
            inputs_gpu = utils_data.make_torch_tensor_and_send_to_device(inputs_cpu, device)
            labels_gpu = utils_data.make_torch_tensor_and_send_to_device(labels_cpu, device)
            labels_gpu_1hot = utils_data.make_label_onehot(labels_gpu, args.num_labels)
            # inputs, labels_one_hot = utils_data.make_torch_tensors_and_send_to_device(inputs, labels, device, args.num_labels)
            dice_score_tr = dice_score_tr + (1 - loss_function(torch.nn.Softmax(dim=1)(model(inputs_gpu)[-1]), labels_gpu_1hot))

        logging.info("Evaluating entire validation dataset...")
        n_batches_vl = images_vl.shape[-1] // args.batch_size
        for iteration in range(n_batches_vl):
            inputs_cpu, labels_cpu = utils_data.get_batch(images_vl, labels_vl, args.batch_size, batch_type = 'sequential', start_idx = iteration * args.batch_size)
            inputs_gpu = utils_data.make_torch_tensor_and_send_to_device(inputs_cpu, device)
            labels_gpu = utils_data.make_torch_tensor_and_send_to_device(labels_cpu, device)
            labels_gpu_1hot = utils_data.make_label_onehot(labels_gpu, args.num_labels)
            dice_score_vl = dice_score_vl + (1 - loss_function(torch.nn.Softmax(dim=1)(model(inputs_gpu)[-1]), labels_gpu_1hot))
    
    # set model back to training mode
    model.train()

    return dice_score_tr / n_batches_tr, dice_score_vl / n_batches_vl

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
    parser.add_argument('--cv_fold_num', default=1, type=int)
    parser.add_argument('--num_labels', default=2, type=int)

    parser.add_argument('--save_path', default='/data/scratch/nkarani/projects/crael/seg/logdir/')
    
    parser.add_argument('--data_aug_prob', default=0.5, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_iterations', default=50001, type=int)
    parser.add_argument('--eval_frequency', default=5000, type=int)
    parser.add_argument('--save_frequency', default=10000, type=int)
    
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
    logging.info("Using device: " + str(device))
    logging.info("Number of devices: " + str(torch.cuda.device_count()))
    # All the available GPUs are being used!
    # It’s natural to execute your forward, backward propagations on multiple GPUs.
    # However, Pytorch will only use one GPU by default.
    # You can easily run your operations on multiple GPUs by making your model run parallelly using DataParallel.

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
    # wandb.tensorboard.patch(root_logdir=summary_path)
    # wandb.init()
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
    # load image and label
    # ===================================
    logging.info('Reading training and validation data')
    data_tr = data_loader.load_data(args, args.sub_dataset, 'train')
    data_vl = data_loader.load_data(args, args.sub_dataset, 'validation')

    images_tr = data_tr["images"]
    labels_tr = data_tr["labels"]
    subject_names_tr = data_tr["subject_names"]

    images_vl = data_vl["images"]
    labels_vl = data_vl["labels"]
    subject_names_vl = data_vl["subject_names"]

    if args.debugging == 1:
        logging.info('training images: ' + str(images_tr.shape))
        logging.info('training labels: ' + str(labels_tr.shape)) # not one hot ... has one channel only
        logging.info(images_tr.dtype)
        logging.info(labels_tr.dtype)
        logging.info(np.min(images_tr))
        logging.info(np.max(labels_tr))
        logging.info(np.unique(labels_tr))
        logging.info('training subject names')
        n_training_images = subject_names_tr.shape[0]
        for n in range(n_training_images):
            logging.info(subject_names_tr[n])
        logging.info('validation images: ' + str(images_vl.shape))
        logging.info('validation labels: ' + str(labels_vl.shape)) # not one hot ... has one channel only
        logging.info('validation subject names')
        n_validation_images = subject_names_vl.shape[0]
        for n in range(n_validation_images):
            logging.info(subject_names_vl[n])

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
    
    # net = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # RuntimeError: CUDA error: out of memory
    # CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
    # For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

    # ===================================
    # define loss
    # https://docs.monai.io/en/stable/losses.html
    #   include_background: if False, channel index 0 (background category) is excluded from the calculation.
    #                       if the non-background segmentations are small compared to the total image size
    #                       they can get overwhelmed by the signal from the background so excluding it in such cases helps convergence.
    # ===================================
    logging.info('Defining losses')
    dice_loss_function = DiceLoss(include_background=False)
    dice_loss_function = dice_loss_function.to(device) 

    if args.method_invariance == 3:
        cons_loss_function = torch.nn.MSELoss()
        cons_loss_function = cons_loss_function.to(device) 

    # ===================================
    # define optimizer
    # ===================================
    logging.info('Defining optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    # ===================================
    # set model to train mode
    # ===================================
    model.train()

    # ===================================
    # keep track of best validation performance 
    # ===================================
    best_dice_score_vl = 0.0

    # ===================================
    # run training iterations
    # ===================================
    logging.info('Starting training iterations')
    for iteration in range(args.max_iterations):

        # # tracking cpu and gpu utilization
        # info0 = nvidia_smi.nvmlDeviceGetMemoryInfo(nvidia_smi.nvmlDeviceGetHandleByIndex(0))
        # info1 = nvidia_smi.nvmlDeviceGetMemoryInfo(nvidia_smi.nvmlDeviceGetHandleByIndex(1))
        # info2 = nvidia_smi.nvmlDeviceGetMemoryInfo(nvidia_smi.nvmlDeviceGetHandleByIndex(2))
        # info3 = nvidia_smi.nvmlDeviceGetMemoryInfo(nvidia_smi.nvmlDeviceGetHandleByIndex(3))
        # usage0 = 100*info0.free/info0.total
        # usage1 = 100*info1.free/info1.total
        # usage2 = 100*info2.free/info2.total
        # usage3 = 100*info3.free/info3.total
        # writer.add_scalar("Utilization/GPU0", usage0, iteration+1)
        # writer.add_scalar("Utilization/GPU1", usage1, iteration+1)
        # writer.add_scalar("Utilization/GPU2", usage2, iteration+1)
        # writer.add_scalar("Utilization/GPU3", usage3, iteration+1)

        if iteration % args.eval_frequency == 0:
            logging.info('Training iteration ' + str(iteration + 1) + '...')

        # ===================================
        # get a batch of original (pre-processed) training images and labels
        # ===================================
        inputs_cpu, labels_cpu = utils_data.get_batch(images_tr, labels_tr, args.batch_size)

        # ===================================
        # do data augmentation / make batches as your method wants them to be
        # ===================================
        if args.method_invariance == 1: # data augmentation
            inputs1_cpu, labels1_cpu = utils_data.transform_for_data_aug(inputs_cpu, labels_cpu, data_aug_prob = args.data_aug_prob)
            if iteration < 5:
                savefilename = vis_path + 'iter' + str(iteration) + '.png'
                utils_vis.save_images_and_labels_orig_and_transformed(inputs_cpu, labels_cpu,
                                                                      inputs1_cpu, labels1_cpu,
                                                                      savefilename)

        elif args.method_invariance == 2 or args.method_invariance == 3: # consistency loss
            inputs1_cpu, inputs2_cpu, labels1_cpu = utils_data.transform_for_data_cons(inputs_cpu, labels_cpu, data_aug_prob = args.data_aug_prob)
            if iteration < 5:
                savefilename = vis_path + 'iter' + str(iteration) + '.png'
                utils_vis.save_images_and_labels_orig_and_transformed_two_ways(inputs_cpu, labels_cpu,
                                                                               inputs1_cpu, labels1_cpu,
                                                                               inputs2_cpu, labels1_cpu,
                                                                               savefilename)

        # ===================================
        # send stuff to gpu
        # ===================================
        inputs_gpu = utils_data.make_torch_tensor_and_send_to_device(inputs_cpu, device)
        labels_gpu = utils_data.make_torch_tensor_and_send_to_device(labels_cpu, device)
        labels_gpu_1hot = utils_data.make_label_onehot(labels_gpu, args.num_labels)
        if args.method_invariance != 0:
            inputs1_gpu = utils_data.make_torch_tensor_and_send_to_device(inputs1_cpu, device)
            labels1_gpu = utils_data.make_torch_tensor_and_send_to_device(labels1_cpu, device)
            labels1_gpu_1hot = utils_data.make_label_onehot(labels1_gpu, args.num_labels)
        if args.method_invariance == 2 or args.method_invariance == 3:
            inputs2_gpu = utils_data.make_torch_tensor_and_send_to_device(inputs2_cpu, device)
                
        # ===================================
        # https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
        # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
        # It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
        # ===================================
        optimizer.zero_grad()

        # ===================================
        # pass through model and compute predictions
        # ===================================
        model_outputs = model(inputs_gpu)
        outputs = model_outputs[-1]
        outputs_probabilities = torch.nn.Softmax(dim=1)(outputs)
        # compute loss value for these predictions
        dice_loss = dice_loss_function(outputs_probabilities, labels_gpu_1hot)
        # log loss to the tensorboard
        writer.add_scalar("TRAINING/DiceLossPerBatch", dice_loss, iteration+1)

        # ===================================
        # add additional regularization losses, according to the chosen method
        # ===================================
        if args.method_invariance == 0: # no regularization
            total_loss = dice_loss
        
        elif args.method_invariance == 1: # data augmentation
        
            model_outputs1 = model(inputs1_gpu)
            outputs1 = model_outputs1[-1]
            outputs1_probabilities = torch.nn.Softmax(dim=1)(outputs1)
        
            # loss on data aug samples
            dice_loss_data_aug = dice_loss_function(outputs1_probabilities, labels1_gpu_1hot)
        
            # total loss
            total_loss = (dice_loss + args.lambda_data_aug * dice_loss_data_aug) / (1 + args.lambda_data_aug)
            writer.add_scalar("TRAINING/DataAugLossPerBatch", dice_loss_data_aug, iteration+1)
        
        elif args.method_invariance == 2: # consistency regularization
        
            model_outputs1 = model(inputs1_gpu)
            outputs1 = model_outputs1[-1]
            outputs1_probabilities = torch.nn.Softmax(dim=1)(outputs1)
            model_outputs2 = model(inputs2_gpu)
            outputs2 = model_outputs2[-1]
            outputs2_probabilities = torch.nn.Softmax(dim=1)(outputs2)
        
            # check if inversion has happened correctly:
            if iteration < 5:
                utils_vis.save_debug(inputs_cpu,
                                     labels_cpu,
                                     inputs1_cpu,
                                     labels1_cpu,
                                     inputs2_cpu,
                                     labels1_cpu,
                                     torch.clone(outputs1_probabilities).detach().cpu().numpy(),
                                     torch.clone(outputs2_probabilities).detach().cpu().numpy(),
                                     vis_path + 'preds_iter' + str(iteration) + '.png')
        
            # consistency loss
            dice_loss_consistency = dice_loss_function(outputs1_probabilities, outputs2_probabilities)
        
            # total loss
            total_loss = (dice_loss + args.lambda_reg_consis * dice_loss_consistency) / (1 + args.lambda_reg_consis)
            writer.add_scalar("TRAINING/ConsistencyLossPerBatch", dice_loss_consistency, iteration+1)
        
        elif args.method_invariance == 3: # consistency regularization at each layer
        
            # make sure you are using a model with heads
            model_outputs1 = model(inputs1_gpu)
            outputs1 = model_outputs1[-1]
            outputs1_probabilities = torch.nn.Softmax(dim=1)(outputs1)
            model_outputs2 = model(inputs2_gpu)
            outputs2 = model_outputs2[-1]
            outputs2_probabilities = torch.nn.Softmax(dim=1)(outputs2)
                
            # get output of heads
            heads1 = model_outputs1[0:-1]
            heads2 = model_outputs2[0:-1]
        
            # vis outputs of heads
            if iteration < 5:
                utils_vis.save_heads(inputs_cpu,
                                     inputs1_cpu,
                                     inputs2_cpu,
                                     torch.clone(heads1[0]).detach().cpu().numpy(),
                                     torch.clone(heads2[0]).detach().cpu().numpy(),
                                     torch.clone(heads1[1]).detach().cpu().numpy(),
                                     torch.clone(heads2[1]).detach().cpu().numpy(),
                                     torch.clone(heads1[2]).detach().cpu().numpy(),
                                     torch.clone(heads2[2]).detach().cpu().numpy(),
                                     torch.clone(outputs1_probabilities).detach().cpu().numpy(),
                                     torch.clone(outputs2_probabilities).detach().cpu().numpy(),
                                     vis_path + 'heads_iter' + str(iteration) + '.png')
            
            # consistency loss at each layer
            cons_l1 = cons_loss_function(heads1[0], heads2[0])
            cons_l2 = cons_loss_function(heads1[1], heads2[1])
            cons_l3 = cons_loss_function(heads1[2], heads2[2])
            cons_l4 = cons_loss_function(heads1[3], heads2[3])
            cons_l5 = cons_loss_function(heads1[4], heads2[4])
            cons_l6 = cons_loss_function(heads1[5], heads2[5])
            cons_l7 = cons_loss_function(heads1[6], heads2[6])
            cons_l8 = cons_loss_function(heads1[7], heads2[7])
            cons_l9 = cons_loss_function(heads1[8], heads2[8])
            
            # weights for the loss at each layer
            alpha1 = args.lambda_consis * ((1 / 9) ** args.alpha_layer)
            alpha2 = args.lambda_consis * ((2 / 9) ** args.alpha_layer)
            alpha3 = args.lambda_consis * ((3 / 9) ** args.alpha_layer)
            alpha4 = args.lambda_consis * ((4 / 9) ** args.alpha_layer)
            alpha5 = args.lambda_consis * ((5 / 9) ** args.alpha_layer)
            alpha6 = args.lambda_consis * ((6 / 9) ** args.alpha_layer)
            alpha7 = args.lambda_consis * ((7 / 9) ** args.alpha_layer)
            alpha8 = args.lambda_consis * ((8 / 9) ** args.alpha_layer)
            alpha9 = args.lambda_consis * ((9 / 9) ** args.alpha_layer)

            # total consistency loss
            loss_consistency = alpha1 * cons_l1 + alpha2 * cons_l2 + alpha3 * cons_l3 + alpha4 * cons_l4 + \
                            alpha5 * cons_l5 + alpha6 * cons_l6 + alpha7 * cons_l7 + alpha8 * cons_l8 + alpha9 * cons_l9
            lambda_consis = alpha1 + alpha2 + alpha3 + alpha4 + alpha5 + alpha6 + alpha7 + alpha8 + alpha9

            # loss on data aug samples
            dice_loss_data_aug = dice_loss_function(outputs1_probabilities, labels1_gpu_1hot)
            
            # total loss
            total_loss = (dice_loss + args.lambda_data_aug * dice_loss_data_aug + lambda_consis * loss_consistency) / (1 + lambda_consis + args.lambda_data_aug)
            
            # tensorboard
            writer.add_scalar("TRAINING/ConsistencyLossPerBatch", loss_consistency, iteration+1)
            writer.add_scalar("TRAINING/DataAugLossPerBatch", dice_loss_data_aug, iteration+1)
        
        writer.add_scalar("TRAINING/TotalLossPerBatch", total_loss, iteration+1)

        # ===================================
        # For the last batch after every some epochs, write images, labels and predictions to TB
        # ===================================
        if iteration % 100 == 0:
            writer.add_figure('Training',
                               utils_vis.show_images_labels_predictions(inputs_gpu, labels_gpu_1hot, outputs_probabilities),
                               global_step = iteration+1)

            if args.method_invariance != 0:
                writer.add_figure('TrainingTransformed1',
                                   utils_vis.show_images_labels_predictions(inputs1_gpu, labels1_gpu_1hot, outputs1_probabilities),
                                   global_step = iteration+1)

            if args.method_invariance == 2 or args.method_invariance == 3:
                writer.add_figure('TrainingTransformed2',
                                   utils_vis.show_images_labels_predictions(inputs2_gpu, labels1_gpu_1hot, outputs2_probabilities),
                                   global_step = iteration+1)
        
        # ===================================
        # https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
        # https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
        # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
        # These are accumulated into x.grad for every parameter x.
        # In pseudo-code: x.grad += dloss/dx
        # ===================================            
        total_loss.backward()
        
        # ===================================
        # https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
        # optimizer.step updates the value of x using the gradient x.grad.
        # For example, the SGD optimizer performs:
        # x += -lr * x.grad
        # ===================================
        optimizer.step()

        # ===================================
        # evaluate on entire training and validation sets every once in a while
        # ===================================
        if iteration % args.eval_frequency == 0:
            dice_score_tr, dice_score_vl = evaluate(args,
                                                    model,
                                                    images_tr,
                                                    labels_tr,
                                                    images_vl,
                                                    labels_vl,
                                                    device,
                                                    dice_loss_function)
            writer.add_scalar("TRAINING/DiceScoreEntireTrainSet", dice_score_tr, iteration+1)
            writer.add_scalar("TRAINING/DiceScoreEntireValSet", dice_score_vl, iteration+1)

            # ===================================
            # save best model so far, according to performance on validation set
            # ===================================
            if best_dice_score_vl <= dice_score_vl:
                best_dice_score_vl = dice_score_vl
                stuff_to_be_saved = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                model_name = 'model_best_dice_iter' + str(iteration) + '.pt'
                torch.save(stuff_to_be_saved, models_path + model_name)
                logging.info('Found new best dice on val set: ' + str(best_dice_score_vl) + ' at iteration ' + str(iteration + 1) + '. Saved model.')

        # ===================================
        # save models at some frequency irrespective of whether this is the best model or not
        # ===================================
        if iteration % args.save_frequency == 0:
            stuff_to_be_saved = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            model_name = 'model_iter' + str(iteration) + '.pt'
            torch.save(stuff_to_be_saved, models_path + model_name)

        # ===================================
        # flush all summaries to tensorboard
        # ===================================
        writer.flush()

    nvidia_smi.nvmlShutdown()