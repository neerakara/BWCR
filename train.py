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
import data_placenta

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
            inputs, labels = utils_data.get_batch(images_tr, labels_tr, args.batch_size, batch_type = 'sequential', start_idx = iteration * args.batch_size)
            inputs, labels_one_hot = utils_data.make_torch_tensors_and_send_to_device(inputs, labels, device, args.num_labels)
            dice_score_tr = dice_score_tr + (1 - loss_function(torch.nn.Softmax(dim=1)(model(inputs)), labels_one_hot))

        logging.info("Evaluating entire validation dataset...")
        n_batches_vl = images_vl.shape[-1] // args.batch_size
        for iteration in range(n_batches_vl):
            inputs, labels = utils_data.get_batch(images_vl, labels_vl, args.batch_size, batch_type = 'sequential', start_idx = iteration * args.batch_size)
            inputs, labels_one_hot = utils_data.make_torch_tensors_and_send_to_device(inputs, labels, device, args.num_labels)
            dice_score_vl = dice_score_vl + (1 - loss_function(torch.nn.Softmax(dim=1)(model(inputs)), labels_one_hot))
    
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
    parser.add_argument('--dataset', default='placenta')
    parser.add_argument('--data_path', default='/data/vision/polina/projects/fetal/projects/placenta-segmentation/data/split-nifti-processed/')
    parser.add_argument('--save_path', default='/data/scratch/nkarani/projects/crael/seg/logdir/')
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--max_iterations', default=21, type=int)#200001
    parser.add_argument('--eval_frequency', default=5000, type=int)
    parser.add_argument('--save_frequency', default=20000, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--data_aug_prob', default=0.5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--cv_fold_num', default=1, type=int)
    parser.add_argument('--run_number', default=1, type=int)
    parser.add_argument('--debugging', default=1, type=int)    
    parser.add_argument('--method_invariance', default=2, type=int) # 0: no reg, 1: data aug, 2: consistency, 3: consistency in each layer
    parser.add_argument('--lambda_reg', default=1.0, type=float) # weight for regularization loss
    parser.add_argument('--alpha_layer', default=1.0, type=float) # growth of regularization loss weight with network depth
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
    # ===================================
    logging_dir = utils_generic.make_expdir(args)

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
    # load image and label
    # ===================================
    logging.info('Reading training data')
    images_tr, labels_tr, img_paths_tr, lbl_paths_tr = data_placenta.load_images_and_labels(args.data_path, train_test_val = 'train', cv_fold = args.cv_fold_num)
    images_vl, labels_vl, img_paths_vl, lbl_paths_vl = data_placenta.load_images_and_labels(args.data_path, train_test_val = 'validation', cv_fold = args.cv_fold_num)
    if args.debugging == 1:
        logging.info('training images: ' + str(images_tr.shape))
        logging.info('training labels: ' + str(labels_tr.shape)) # not one hot ... has one channel only
        logging.info(images_tr.dtype)
        logging.info(labels_tr.dtype)
        logging.info(np.min(images_tr))
        logging.info(np.max(labels_tr))
        logging.info(np.unique(labels_tr))
        logging.info('training images')
        for img_path in img_paths_tr:
            logging.info(img_path[img_path.rfind('/'):])
        logging.info('validation images: ' + str(images_vl.shape))
        logging.info('validation labels: ' + str(labels_vl.shape)) # not one hot ... has one channel only
        logging.info('validation images')
        for img_path in img_paths_vl:
            logging.info(img_path[img_path.rfind('/'):])

    # ===================================
    # define model
    # ===================================
    logging.info('Defining segmentation model')
    model = models.UNet2d(in_channels = 1,
                          num_labels = args.num_labels,
                          squeeze = False)
    model = model.to(device)
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

        if iteration % args.eval_frequency == 0:
            logging.info('Training iteration ' + str(iteration + 1) + '...')

        # ===================================
        # get a batch of original (pre-processed) training images and labels
        # ===================================
        inputs, labels = utils_data.get_batch(images_tr, labels_tr, args.batch_size)
        # training loss goes to zero when trained on only this batch.
        # inputs, labels = utils_data.get_batch(images_tr, labels_tr, args.batch_size, 'sequential', 120)
        if args.debugging == 1:
            num_fg = utils_data.get_number_of_frames_with_fg(labels)
            writer.add_scalar("TRAINING/Num_FG_slices", num_fg, iteration+1)

        # ===================================
        # do data augmentation / make batches as your method wants them to be
        # ===================================
        if args.method_invariance == 1: # data augmentation
            inputs1, labels1, geom_params1 = utils_data.transform_images_and_labels(inputs, labels, data_aug_prob = args.data_aug_prob)
            if iteration < 5:
                savefilename = vis_path + 'iter' + str(iteration) + '.png'
                utils_vis.save_images_and_labels_orig_and_transformed(inputs, labels, inputs1, labels1, savefilename)
            inputs1, labels1_one_hot = utils_data.make_torch_tensors_and_send_to_device(inputs1, labels1, device, args.num_labels)

        elif args.method_invariance == 2: # consistency loss
            inputs1, labels1, geom_params1 = utils_data.transform_images_and_labels(inputs, labels, data_aug_prob = args.data_aug_prob)
            inputs2, labels2, geom_params2 = utils_data.transform_images_and_labels(inputs, labels, data_aug_prob = args.data_aug_prob)
            if iteration < 5:
                savefilename = vis_path + 'iter' + str(iteration) + '.png'
                utils_vis.save_images_and_labels_orig_and_transformed_two_ways(inputs, labels, inputs1, labels1, inputs2, labels2, savefilename)
            inputs1, labels1_one_hot = utils_data.make_torch_tensors_and_send_to_device(inputs1, labels1, device, args.num_labels)
            inputs2, labels2_one_hot = utils_data.make_torch_tensors_and_send_to_device(inputs2, labels2, device, args.num_labels)

        inputs, labels_one_hot = utils_data.make_torch_tensors_and_send_to_device(inputs, labels, device, args.num_labels)
        
        # for consistency losses (whether at the end or at all layers)
        # call transform twice
        # inputs1, labels1, tx1, ty1, theta1, sc1 = utils_data.transform_images_and_labels(inputs, labels, data_aug_prob = args.data_aug_prob)
        # inputs2, labels2, tx2, ty2, theta2, sc2 = utils_data.transform_images_and_labels(inputs, labels, data_aug_prob = args.data_aug_prob)
        # cons_loss = cons_loss_function(model(inputs1), model(inputs2), tx1, ty1, theta1, sc1, tx2, ty2, theta2, sc2)

        # for implementing layer-wise losses, understand how the 'forward' function in the model works.
        # currenly, it seems to be running when outputs = model(inputs) is called
        # figure out how to query multiple layer features from it
        
        # ===================================
        # https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
        # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
        # It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
        # ===================================
        optimizer.zero_grad()

        # ===================================
        # pass through model and compute predictions
        # ===================================
        outputs = model(inputs)
        outputs_probabilities = torch.nn.Softmax(dim=1)(outputs)
        # compute loss value for these predictions
        dice_loss = dice_loss_function(outputs_probabilities, labels_one_hot)
        # log loss to the tensorboard
        writer.add_scalar("TRAINING/DiceLossPerBatch", dice_loss, iteration+1)

        # ===================================
        # add additional regularization losses, according to the chosen method
        # ===================================
        if args.method_invariance == 0: # no regularization
            total_loss = dice_loss
        elif args.method_invariance == 1: # data augmentation
            outputs1 = model(inputs1)
            outputs1_probabilities = torch.nn.Softmax(dim=1)(outputs1)
            # loss on data aug samples
            dice_loss_data_aug = dice_loss_function(outputs1_probabilities, labels1_one_hot)
            # total loss
            total_loss = (dice_loss + args.lambda_reg * dice_loss_data_aug) / (1 + args.lambda_reg)
            writer.add_scalar("TRAINING/DataAugLossPerBatch", dice_loss_data_aug, iteration+1)
        elif args.method_invariance == 2: # consistency regularization
            outputs1 = model(inputs1)
            outputs1_probabilities = torch.nn.Softmax(dim=1)(outputs1)
            outputs2 = model(inputs2)
            outputs2_probabilities = torch.nn.Softmax(dim=1)(outputs2)
            # invert geometric params
            outputs1_probabilities_inv = utils_data.invert_geometric_transforms(outputs1_probabilities, geom_params1)
            outputs2_probabilities_inv = utils_data.invert_geometric_transforms(outputs2_probabilities, geom_params2)
            # check if inversion has happened correctly:
            if iteration < 5:
                utils_vis.save_debug(np.copy(inputs.cpu().numpy()),
                                     np.copy(labels),
                                     np.copy(inputs1.cpu().numpy()),
                                     np.copy(labels1),
                                     np.copy(inputs2.cpu().numpy()),
                                     np.copy(labels2),
                                     torch.clone(outputs1_probabilities).detach().cpu().numpy(),
                                     torch.clone(outputs1_probabilities_inv).detach().cpu().numpy(),
                                     torch.clone(outputs2_probabilities).detach().cpu().numpy(),
                                     torch.clone(outputs2_probabilities_inv).detach().cpu().numpy(),
                                     vis_path + 'preds_iter' + str(iteration) + '.png')
            # consistency loss
            dice_loss_consistency = dice_loss_function(outputs1_probabilities_inv, outputs2_probabilities_inv)
            # total loss
            total_loss = (dice_loss + args.lambda_reg * dice_loss_consistency) / (1 + args.lambda_reg)
            writer.add_scalar("TRAINING/ConsistencyLossPerBatch", dice_loss_consistency, iteration+1)
        writer.add_scalar("TRAINING/TotalLossPerBatch", total_loss, iteration+1)

        # ===================================
        # For the last batch after every some epochs, write images, labels and predictions to TB
        # ===================================
        if iteration % 100 == 0:
            writer.add_figure('Training',
                               utils_vis.show_images_labels_predictions(inputs, labels_one_hot, outputs_probabilities),
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