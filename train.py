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
            inputs_cpu, labels_cpu = utils_data.get_batch(images_tr,
                                                          labels_tr,
                                                          args.batch_size,
                                                          batch_type = 'sequential',
                                                          start_idx = iteration * args.batch_size)
            inputs_gpu = utils_data.torch_and_send_to_device(inputs_cpu, device)
            labels_gpu = utils_data.torch_and_send_to_device(labels_cpu, device)
            labels_gpu_1hot = utils_data.make_label_onehot(labels_gpu, args.num_labels)
            # inputs, labels_one_hot = utils_data.make_torch_tensors_and_send_to_device(inputs, labels, device, args.num_labels)
            dice_score_tr = dice_score_tr + (1 - loss_function(torch.nn.Softmax(dim=1)(model(inputs_gpu)[-1]), labels_gpu_1hot))

        logging.info("Evaluating entire validation dataset...")
        n_batches_vl = images_vl.shape[-1] // args.batch_size
        for iteration in range(n_batches_vl):
            inputs_cpu, labels_cpu = utils_data.get_batch(images_vl,
                                                          labels_vl,
                                                          args.batch_size,
                                                          batch_type = 'sequential',
                                                          start_idx = iteration * args.batch_size)
            inputs_gpu = utils_data.torch_and_send_to_device(inputs_cpu, device)
            labels_gpu = utils_data.torch_and_send_to_device(labels_cpu, device)
            labels_gpu_1hot = utils_data.make_label_onehot(labels_gpu, args.num_labels)
            dice_score_vl = dice_score_vl + (1 - loss_function(torch.nn.Softmax(dim=1)(model(inputs_gpu)[-1]), labels_gpu_1hot))
    
    # set model back to training mode
    model.train()

    return dice_score_tr / n_batches_tr, dice_score_vl / n_batches_vl

# ==========================================
# ==========================================
def get_probs_and_outputs(model, inputs_gpu):
    model_outputs = model(inputs_gpu)
    outputs_probs = torch.nn.Softmax(dim=1)(model_outputs[-1])
    return model_outputs, outputs_probs

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
    
    parser.add_argument('--dataset', default='placenta') # placenta / prostate
    parser.add_argument('--sub_dataset', default='RUNMC') # prostate: BIDMC / BMC / HK / I2CVB / RUNMC / UCL
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
    
    parser.add_argument('--model_has_heads', default=0, type=int)    
    parser.add_argument('--method_invariance', default=0, type=int) # 0: no reg, 1: data aug, 2: consistency in each layer (geom + int), 3: consistency in each layer (int)
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
    if args.method_invariance in [2, 3]:
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

        if iteration % args.log_frequency == 0:
            logging.info('Training iteration ' + str(iteration + 1) + '...')

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

        # ===================================
        # https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
        # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
        # It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
        # ===================================
        optimizer.zero_grad()

        # ===================================
        # get a batch of original (pre-processed) training images and labels
        # ===================================
        inputs_cpu, labels_cpu = utils_data.get_batch(images_tr, labels_tr, args.batch_size)
        inputs_gpu = utils_data.torch_and_send_to_device(inputs_cpu, device)
        labels_gpu = utils_data.torch_and_send_to_device(labels_cpu, device)
        labels_gpu_1hot = utils_data.make_label_onehot(labels_gpu, args.num_labels)
        if iteration < 5:
            utils_vis.save_images_and_labels(inputs_gpu, labels_gpu, vis_path + 'orig_iter' + str(iteration) + '.png')       
        # pass through model and compute predictions
        model_outputs, outputs_probs = get_probs_and_outputs(model, inputs_gpu)
        # compute loss value for these predictions
        dice_loss = dice_loss_function(outputs_probs, labels_gpu_1hot)
        # log loss to the tensorboard
        writer.add_scalar("TRAINING/DiceLossPerBatch", dice_loss, iteration+1)

        # ===================================
        # add additional regularization losses, according to the chosen method
        # ===================================
        if args.method_invariance == 0: # no regularization
            total_loss = dice_loss
        
        elif args.method_invariance == 1: # data augmentation
            
            # transform the batch
            inputs1_gpu, labels1_gpu, geom_params1 = utils_data.transform_batch(inputs_gpu,
                                                                                labels_gpu,
                                                                                data_aug_prob = args.data_aug_prob,
                                                                                device = device)
            # visualize training samples
            if iteration < 5:
                utils_vis.save_images_and_labels(inputs1_gpu,
                                                 labels1_gpu,
                                                 vis_path + 't1_iter' + str(iteration) + '.png')

            # convert labels to 1hot
            labels1_gpu_1hot = utils_data.make_label_onehot(labels1_gpu,
                                                            args.num_labels)

            # compute predictions for the transformed batch
            model_outputs1, outputs_probs1 = get_probs_and_outputs(model,
                                                                   inputs1_gpu)

            # loss on data aug samples
            dice_loss_data_aug = dice_loss_function(outputs_probs1,
                                                    labels1_gpu_1hot)

            # total loss
            total_loss = dice_loss_data_aug
            writer.add_scalar("TRAINING/DataAugLossPerBatch", dice_loss_data_aug, iteration+1)
        
        elif args.method_invariance in [2, 3]: # consistency regularization at each layer

            # transform the batch
            inputs1_gpu, labels1_gpu, t1 = utils_data.transform_batch(inputs_gpu,
                                                                      labels_gpu,
                                                                      data_aug_prob = args.data_aug_prob,
                                                                      device = device)

            # apply same geometric transform on both batches
            if args.method_invariance == 3:
                inputs2_gpu, labels2_gpu, t2 = utils_data.transform_batch(inputs_gpu,
                                                                          labels_gpu,
                                                                          data_aug_prob = args.data_aug_prob,
                                                                          device = device,
                                                                          t = t1)
            # apply different geometric transforms on both batches
            elif args.method_invariance == 2:
                inputs2_gpu, labels2_gpu, t2 = utils_data.transform_batch(inputs_gpu,
                                                                          labels_gpu,
                                                                          data_aug_prob = args.data_aug_prob,
                                                                          device = device)

            # visualize training samples
            if iteration < 5:
                utils_vis.save_images_and_labels(inputs1_gpu,
                                                 labels1_gpu,
                                                 vis_path + 't1_iter' + str(iteration) + '.png')
                utils_vis.save_images_and_labels(inputs2_gpu,
                                                 labels2_gpu,
                                                 vis_path + 't2_iter' + str(iteration) + '.png')

            # convert labels to 1hot
            labels1_gpu_1hot = utils_data.make_label_onehot(labels1_gpu,
                                                            args.num_labels)
            labels2_gpu_1hot = utils_data.make_label_onehot(labels2_gpu,
                                                            args.num_labels)
            
            # compute predictions for both the transformed batches
            model_outputs1, outputs_probs1 = get_probs_and_outputs(model,
                                                                   inputs1_gpu)
            model_outputs2, outputs_probs2 = get_probs_and_outputs(model,
                                                                   inputs2_gpu)        

            # loss on data aug samples (one of the two transformations)
            dice_loss_data_aug = dice_loss_function(outputs_probs1, labels1_gpu_1hot)
            writer.add_scalar("TRAINING/DataAugLossPerBatch", dice_loss_data_aug, iteration+1)

            # consistency loss at each layer
            cons_loss_layer_l = []
            weight_layer_l = []
            num_layers = len(model_outputs1)
            
            for l in range(num_layers):
            
                if args.method_invariance == 3:
                    # same geometric transform on both batches, so no need to invert
                    cons_loss_layer_l.append(cons_loss_function(model_outputs1[l],
                                                                model_outputs2[l]))
            
                elif args.method_invariance == 2:
                    # different geometric transforms on both batches, so need to invert
                    cons_loss_layer_l.append(cons_loss_function(utils_data.invert_geometric_transforms(model_outputs1[l], t1),
                                                                utils_data.invert_geometric_transforms(model_outputs2[l], t2)))
            
                weight_layer_l.append(args.lambda_consis * (((l+1) / num_layers) ** args.alpha_layer))
            
                writer.add_scalar("TRAINING/ConsistencyLossPerBatchLayer"+str(l+1), cons_loss_layer_l[l], iteration+1)
                writer.add_scalar("TRAINING/ConsistencyLossWeightLayer"+str(l+1), weight_layer_l[l], iteration+1)
            
            # total consistency loss
            loss_consistency = sum(i[0] * i[1] for i in zip(weight_layer_l, cons_loss_layer_l))
            writer.add_scalar("TRAINING/ConsistencyLossPerBatch", loss_consistency, iteration+1)
            
            # total loss
            total_loss = (dice_loss_data_aug + loss_consistency) / (1 + sum(weight_layer_l))
        
        # total loss to tensorboard
        writer.add_scalar("TRAINING/TotalLossPerBatch", total_loss, iteration+1)

        # ===================================
        # For the last batch after every some epochs, write images, labels and predictions to TB
        # ===================================
        if iteration % 100 == 0:
            writer.add_figure('Training',
                               utils_vis.show_images_labels_predictions(inputs_gpu, labels_gpu_1hot, outputs_probs),
                               global_step = iteration+1)

            if args.method_invariance != 0:
                writer.add_figure('TrainingTransformed1',
                                   utils_vis.show_images_labels_predictions(inputs1_gpu, labels1_gpu_1hot, outputs_probs1),
                                   global_step = iteration+1)

            if args.method_invariance in [2, 3]:
                writer.add_figure('TrainingTransformed2',
                                   utils_vis.show_images_labels_predictions(inputs2_gpu, labels2_gpu_1hot, outputs_probs2),
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