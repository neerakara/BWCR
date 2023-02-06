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
from monai.losses.ssim_loss import SSIMLoss
# https://docs.monai.io/en/stable/losses.html#monai.losses.ssim_loss.SSIMLoss
from torch.utils.tensorboard import SummaryWriter
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
    dice_scores = utils_data.evaluate(args.dataset, subdataset, args.cv_fold_num, ttv, model, device)

    # set model back to training mode
    model.train()

    return dice_scores

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
    
    parser.add_argument('--dataset', default='prostate') # placenta / prostate / ms
    parser.add_argument('--sub_dataset', default='RUNMC') # prostate: BIDMC / BMC / HK / I2CVB / RUNMC / UCL / InD / OoD
    parser.add_argument('--cv_fold_num', default=1, type=int)
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--save_path', default='/data/scratch/nkarani/projects/crael/seg/logdir/v6/')
    
    parser.add_argument('--data_aug_prob', default=0.5, type=float)
    parser.add_argument('--optimizer', default='adam') # adam / sgd
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_schedule', default=2, type=int)
    parser.add_argument('--lr_schedule_step', default=15000, type=int)
    parser.add_argument('--lr_schedule_gamma', default=0.1, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_iterations', default=50001, type=int)
    parser.add_argument('--log_frequency', default=500, type=int)
    parser.add_argument('--eval_frequency_tr', default=1000, type=int)
    parser.add_argument('--eval_frequency_vl', default=100, type=int)
    parser.add_argument('--save_frequency', default=10000, type=int)
    
    parser.add_argument('--model_has_heads', default=1, type=int)    
    parser.add_argument('--method_invariance', default=2, type=int) # 0: no reg, 1: data aug, 2: consistency in each layer (geom + int), 3: consistency in each layer (int)
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

    images_tr = data_tr["images"]
    labels_tr = data_tr["labels"]
    subject_names_tr = data_tr["subject_names"]

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
    # Create a dictionary to store the EMA weights
    # ===================================
    # ema_weights = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    # Set the decay rate
    # decay = 0.99
    
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
    if args.method_invariance in [2, 3, 20, 30, 200, 300]:
        cons_loss_function = torch.nn.MSELoss(reduction='sum') # sum of squared errors to remove magnitude dependence on image size
        cons_loss_function = cons_loss_function.to(device) 

    # ===================================
    # define optimizer
    # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    # ===================================
    logging.info('Defining optimizer')
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.lr_schedule == 1:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_schedule_step, gamma=args.lr_schedule_gamma)
    elif args.lr_schedule == 2: # in case you need to run for more than max iterations, lr will remain constant at 1e-8
        if args.load_model_num != 0:
            lambda1 = lambda iteration: 1e-7 / args.lr
        else:
            lambda1 = lambda iteration: 1 - ((args.lr - 1e-7) * iteration / (args.max_iterations * args.lr))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # ===================================
    # load pre-trained model
    # ===================================
    if args.load_model_num != 0:
        modelpath = models_path + 'model_iter' + str(args.load_model_num) + '.pt'
        logging.info('loading model weights from: ')
        logging.info(modelpath)
        model.load_state_dict(torch.load(modelpath)['state_dict'])

    # ===================================
    # set model to train mode
    # ===================================
    model.train()

    # ===================================
    # keep track of best validation performance 
    # ===================================
    best_dice_score_vl = 0.0
    # best_dice_score_vl_ema = 0.0

    # ===================================
    # run training iterations
    # ===================================
    logging.info('Starting training iterations')
    for iteration in range(args.load_model_num, args.max_iterations):

        if iteration % args.log_frequency == 0:
            logging.info('Training iteration ' + str(iteration + 1) + '...')

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
        writer.add_scalar("Tr/DiceLossBatch", dice_loss, iteration+1)

        # ===================================
        # additional regularization losses, according to the chosen method
        # ===================================

        # ===================================
        # Method 0: no regularization
        # ===================================
        if args.method_invariance == 0:
            total_loss = dice_loss
        
        # ===================================
        # Method 1: data augmentation
        # ===================================
        elif args.method_invariance in [1, 10, 100]:
            
            # transform the batch
            inputs1_gpu, labels1_gpu, _ = utils_data.transform_batch(inputs_gpu,
                                                                     labels_gpu,
                                                                     data_aug_prob = args.data_aug_prob,
                                                                     device = device)
            # visualize training samples
            if iteration < 5:
                utils_vis.save_images_and_labels(inputs1_gpu, labels1_gpu, vis_path + 't1_iter' + str(iteration) + '.png')

            # convert labels to 1hot
            labels1_gpu_1hot = utils_data.make_label_onehot(labels1_gpu, args.num_labels)

            # compute predictions for the transformed batch
            model_outputs1, outputs_probs1 = get_probs_and_outputs(model, inputs1_gpu)

            # loss on data aug samples
            dice_loss_data_aug = dice_loss_function(outputs_probs1, labels1_gpu_1hot)

            writer.add_scalar("Tr/DataAugLossBatch", dice_loss_data_aug, iteration+1)

            # =======================
            # total loss |  typically, data augmentation is implemented like this
            # =======================
            if args.method_invariance == 1:
                total_loss = dice_loss_data_aug

            # =======================
            # total loss | in the initial implementation, I had implemented data augmentation like this
            # =======================
            elif args.method_invariance == 10:
                total_loss = dice_loss + args.lambda_dataaug * dice_loss_data_aug

            # =======================
            # total loss | We should divide by the total coeff to keep the loss scales comparable across methods
            # =======================
            elif args.method_invariance == 100:
                total_loss = (dice_loss + args.lambda_dataaug * dice_loss_data_aug) / (1 + args.lambda_dataaug)
        
        # ===================================
        # Methods 2 / 3: consistency regularization applied in two different ways
        # Method 2: Most intuitive way to apply the idea. Requires two different geometric transforms to be applied to each batch and all features to be upsampled and inverted.
        # Method 3: Same geometric transform is applied to each batch, so that consistency loss can be computed without inverting the transforms.
        # Two different intensity transforms are applied to the batch in both methods.
        # ===================================
        elif args.method_invariance in [2, 3, 20, 30, 200, 300]: # consistency regularization at each layer

            # transform the batch
            inputs1_gpu, labels1_gpu, t1 = utils_data.transform_batch(inputs_gpu,
                                                                      labels_gpu,
                                                                      data_aug_prob = args.data_aug_prob,
                                                                      device = device)

            # apply same geometric transform on both batches
            if args.method_invariance in [3, 30, 300]:
                inputs2_gpu, labels2_gpu, t2 = utils_data.transform_batch(inputs_gpu,
                                                                          labels_gpu,
                                                                          data_aug_prob = args.data_aug_prob,
                                                                          device = device,
                                                                          t = t1)
            # apply different geometric transforms on both batches
            elif args.method_invariance in [2, 20, 200]:
                inputs2_gpu, labels2_gpu, t2 = utils_data.transform_batch(inputs_gpu,
                                                                          labels_gpu,
                                                                          data_aug_prob = args.data_aug_prob,
                                                                          device = device)

            # visualize training samples
            if iteration < 10:
                utils_vis.save_images_and_labels(inputs1_gpu, labels1_gpu, vis_path + 't1_iter' + str(iteration) + '.png')
                utils_vis.save_images_and_labels(inputs2_gpu, labels2_gpu, vis_path + 't2_iter' + str(iteration) + '.png')

            # convert labels to 1hot
            labels1_gpu_1hot = utils_data.make_label_onehot(labels1_gpu, args.num_labels)
            labels2_gpu_1hot = utils_data.make_label_onehot(labels2_gpu, args.num_labels)
            
            # compute predictions for both the transformed batches
            model_outputs1, outputs_probs1 = get_probs_and_outputs(model, inputs1_gpu)
            model_outputs2, outputs_probs2 = get_probs_and_outputs(model, inputs2_gpu)        

            # loss on data aug samples (one of the two transformations)
            dice_loss_data_aug = dice_loss_function(outputs_probs1, labels1_gpu_1hot)
            writer.add_scalar("Tr/DataAugLossBatch", dice_loss_data_aug, iteration+1)

            # consistency loss at each layer
            cons_loss_layer_l = []
            weight_layer_l = []
            num_layers = len(model_outputs1)

            if args.method_invariance in [2, 20, 200]:
                # generate mask to compute loss, excluding pixels introduced by the geometric transforms
                mask1_gpu = utils_data.apply_geometric_transforms_mask(torch.ones(inputs1_gpu.shape).to(device, dtype = torch.float), t1)
                mask2_gpu = utils_data.apply_geometric_transforms_mask(torch.ones(inputs2_gpu.shape).to(device, dtype = torch.float), t2)
            
            for l in range(num_layers):

                if args.method_invariance in [3, 30, 300]:
                    # same geometric transform on both batches, so no need to invert
                    if args.consis_loss == 1:
                        cons_loss_layer_l.append(cons_loss_function(model_outputs1[l], model_outputs2[l]))
                    elif args.consis_loss == 2: # mse between normalized features
                        cons_loss_layer_l.append(cons_loss_function(torch.nn.functional.normalize(model_outputs1[l], dim=(1,2,3)),
                                                                    torch.nn.functional.normalize(model_outputs2[l], dim=(1,2,3))))
            
                elif args.method_invariance in [2, 20, 200]:
                    # different geometric transforms on both batches, so need to invert
                    features1_inv = utils_data.invert_geometric_transforms(model_outputs1[l], t1)
                    features2_inv = utils_data.invert_geometric_transforms(model_outputs2[l], t2)
                    mask1_resized = utils_data.rescale_tensor(mask1_gpu, features1_inv.shape[-1])
                    mask2_resized = utils_data.rescale_tensor(mask2_gpu, features2_inv.shape[-1])
                    mask1_gpu_inv = utils_data.invert_geometric_transforms_mask(mask1_resized, t1)
                    mask2_gpu_inv = utils_data.invert_geometric_transforms_mask(mask2_resized, t2)
                    mask = torch.mul(mask1_gpu_inv, mask2_gpu_inv)
                    mask = mask.repeat(1,features1_inv.shape[1],1,1)
                    features1_inv_masked = torch.mul(features1_inv, mask)
                    features2_inv_masked = torch.mul(features2_inv, mask)

                    if args.consis_loss == 1:
                        cons_loss_layer_l.append(cons_loss_function(features1_inv_masked, features2_inv_masked))
                    elif args.consis_loss == 2: # mse between normalized features
                        cons_loss_layer_l.append(cons_loss_function(torch.nn.functional.normalize(features1_inv_masked, dim=(1,2,3)),
                                                                    torch.nn.functional.normalize(features2_inv_masked, dim=(1,2,3))))

                    if iteration % 5000 == 0:
                        utils_vis.save_from_list([inputs1_gpu,
                                                  inputs2_gpu,
                                                  features1_inv_masked,
                                                  features2_inv_masked,
                                                  features1_inv_masked - features2_inv_masked],
                                                  vis_path + 'feats_inv_correctly_iter' + str(iteration) + '_l' + str(l+1) + '.png')
            
                weight_layer_l.append((((l+1) / num_layers) ** args.alpha_layer))
                writer.add_scalar("Tr/ConsisLossBatchLayer"+str(l+1), cons_loss_layer_l[l], iteration+1)

            total_weight = sum(weight_layer_l)
            for l in range(num_layers):    
                weight_layer_l[l] = args.lambda_consis * weight_layer_l[l] / total_weight
                writer.add_scalar("Tr/ConsisLossWeightLayer"+str(l+1), weight_layer_l[l], iteration+1)
            
            # total consistency loss
            loss_consistency = sum(i[0] * i[1] for i in zip(weight_layer_l, cons_loss_layer_l))
            writer.add_scalar("Tr/ConsisLossBatch", loss_consistency, iteration+1)
            
            # total loss
            if args.method_invariance in [2, 3]:                
                total_loss = (dice_loss_data_aug + loss_consistency) / (1 + sum(weight_layer_l))
            elif args.method_invariance in [20, 30]:
                total_loss = (dice_loss + loss_consistency) / (1 + sum(weight_layer_l))
            elif args.method_invariance in [200, 300]:
                total_loss = (dice_loss + args.lambda_dataaug * dice_loss_data_aug + loss_consistency) / (1 + args.lambda_dataaug + sum(weight_layer_l))
        
        # total loss to tensorboard
        writer.add_scalar("Tr/TotalLossBatch", total_loss, iteration+1)

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

            if args.method_invariance in [2, 3, 20, 30, 200, 300]:
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
        if args.lr_schedule != 0:
            scheduler.step()
            writer.add_scalar("Tr/lr", scheduler.get_last_lr()[0], iteration+1)

        # # ===================================
        # # Update the EMA weights
        # # ===================================
        # with torch.no_grad():
        #     for name, param in model.named_parameters():
        #         if 'bn' in name:
        #             # Update the running mean and variance of the batch normalization layers
        #             param.update_parameters(decay)
        #         else:
        #             ema_weights[name] = ema_weights[name] * decay + param * (1 - decay)

        # ===================================
        # evaluate on entire validation and test sets every once in a while
        # ===================================
        if (iteration % args.eval_frequency_vl == 0) or (iteration > (args.max_iterations - 100)):
            dice_score_vl = evaluate(args, model, device, args.sub_dataset, 'validation')
            writer.add_scalar("Tr/Dice_Val", np.mean(dice_score_vl), iteration+1)

            # dice_score_vl_ema = evaluate(args, ema.shadow, device, args.sub_dataset, 'validation')
            # writer.add_scalar("Tr/Dice_Val_EMA", np.mean(dice_score_vl_ema), iteration+1)

            dice_score_ts = evaluate(args, model, device, args.sub_dataset, 'test')
            writer.add_scalar("Tr/Dice_Test", np.mean(dice_score_ts), iteration+1)

            # dice_score_ts_ema = evaluate(args, ema.shadow, device, args.sub_dataset, 'test')
            # writer.add_scalar("Tr/Dice_Test_EMA", np.mean(dice_score_ts_ema), iteration+1)

            # writer.add_scalar("Tr/Dice_UCL", np.mean(evaluate(args, model, device, 'UCL', 'test')), iteration+1)
            # writer.add_scalar("Tr/Dice_UCL_EMA", np.mean(evaluate(args, ema.shadow, device, 'UCL', 'test')), iteration+1)
            # writer.add_scalar("Tr/Dice_HK", np.mean(evaluate(args, model, device, 'HK', 'test')), iteration+1)
            # writer.add_scalar("Tr/Dice_HK_EMA", np.mean(evaluate(args, ema.shadow, device, 'HK', 'test')), iteration+1)
            # writer.add_scalar("Tr/Dice_BIDMC", np.mean(evaluate(args, model, device, 'BIDMC', 'test')), iteration+1)
            # writer.add_scalar("Tr/Dice_BIDMC_EMA", np.mean(evaluate(args, ema.shadow, device, 'BIDMC', 'test')), iteration+1)

            # logging.info(iteration)
            # dice_score_ts = np.reshape(dice_score_ts, [1,-1])
            # if iteration == args.load_model_num:
            #     dice_scores_all_iters = dice_score_ts
            # else:
            #     dice_scores_all_iters = np.concatenate((dice_scores_all_iters, dice_score_ts))

            # ===================================
            # save best model so far, according to performance on validation set
            # ===================================
            if (best_dice_score_vl <= np.mean(dice_score_vl)):
                best_dice_score_vl = np.mean(dice_score_vl)
                stuff_to_be_saved = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                model_name = 'best_val_iter' + str(iteration) + '.pt'
                torch.save(stuff_to_be_saved, models_path + model_name)
                logging.info('Found new best dice on val set: ' + str(best_dice_score_vl) + ' at iteration ' + str(iteration + 1) + '. Saved model.')

            # ===================================
            # save best model so far, according to performance on validation set (evaluated on ema weights)
            # ===================================
            # if (best_dice_score_vl_ema <= np.mean(dice_score_vl_ema)):
            #     best_dice_score_vl_ema = np.mean(dice_score_vl_ema)
            #     stuff_to_be_saved = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            #     model_name = 'best_val_ema_iter' + str(iteration) + '.pt'
            #     torch.save(stuff_to_be_saved, models_path + model_name)
            #     logging.info('Found new best dice on val set: ' + str(best_dice_score_vl_ema) + ' at iteration ' + str(iteration + 1) + '. Saved model.')

        # ===================================
        # save models at some frequency irrespective of whether this is the best model or not
        # ===================================
        if iteration % args.eval_frequency_tr == 0:
            
            logging.info("Evaluating entire training dataset...")
            dice_score_tr = evaluate(args, model, device, args.sub_dataset, 'train')
            writer.add_scalar("Tr/DiceTrain", np.mean(dice_score_tr), iteration+1)

            logging.info("saving current model")
            stuff_to_be_saved = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            model_name = 'model_iter' + str(iteration) + '.pt'
            torch.save(stuff_to_be_saved, models_path + model_name)

        # ===================================
        # flush all summaries to tensorboard
        # ===================================
        writer.flush()
    
    # ================
    # save
    # ================
    # np.save(models_path + 'variance_across_training_iters.npy', dice_scores_all_iters)