import os
import numpy as np

# ======================================================
# ======================================================
def make_expdir_2(args):

    # data related
    if args.dataset == 'placenta':
        logdir = args.save_path + args.dataset + '/cv' + str(args.cv_fold_num) + '/' 
    elif args.dataset == 'prostate':
        logdir = args.save_path + args.dataset + '/' + args.sub_dataset + '/cv' + str(args.cv_fold_num) + '/'
    elif args.dataset == 'ms':
        logdir = args.save_path + args.dataset + '/'
    elif args.dataset == 'acdc':
        logdir = args.save_path + args.dataset + '/cv' + str(args.cv_fold_num) + '/' 

    # optimization related
    logdir = logdir + 'lr' + str(args.lr) + '_bs' + str(args.batch_size) + '/' 

    # invariance method
    logdir = logdir + 'l0_' + str(args.l0) 
    logdir = logdir + '/l1_' + args.l1_loss + '_' + str(args.l1) 
    logdir = logdir + '/l2_' + args.l2_loss + '_' + str(args.l2) 
    
    if args.weigh_lambda_con == 1:
        logdir = logdir + '_distance_weighing'

    if args.l2_loss == 'l2_margin':
        logdir = logdir + '/' + str(args.l2_loss_margin)
    
    if args.l2_loss == 'l2_all':
        logdir = logdir + '/alpha' + str(args.alpha_layer)
    
    if args.teacher != 'self':
        logdir = logdir + '/teacher_' + args.teacher
    
    logdir = logdir + '/t_' + str(args.temp) + '/'

    # run number
    if args.out_layer_type == 2:
        logdir = logdir + 'run' + str(args.run_number) + '/'
    else:
        logdir = logdir + 'run' + str(args.run_number) + '_output_layer_type_1/'
    
    return logdir

# ======================================================
# ======================================================
def make_expdir(args):

    # data related
    if args.dataset == 'placenta':
        logdir = args.save_path + args.dataset + '/cv' + str(args.cv_fold_num) + '/' 
    elif args.dataset == 'prostate':
        logdir = args.save_path + args.dataset + '/' + args.sub_dataset + '/cv' + str(args.cv_fold_num) + '/'
    elif args.dataset == 'ms':
        logdir = args.save_path + args.dataset + '/'
    elif args.dataset == 'acdc':
        logdir = args.save_path + args.dataset + '/cv' + str(args.cv_fold_num) + '/' 

    # optimization related
    if args.optimizer == 'sgd':
        logdir = logdir + args.optimizer + '_lr' + str(args.lr) + '_sch' + str(args.lr_schedule) + '_bs' + str(args.batch_size) + '/' 
    else:
        logdir = logdir + 'lr' + str(args.lr) + '_sch' + str(args.lr_schedule) + '_bs' + str(args.batch_size) + '/' 

    # model related
    if args.model_has_heads == 1:
        logdir = logdir + 'unet_with_heads/'
    else:
        logdir = logdir + 'unet/'

    # invariance method
    logdir = logdir + 'm' + str(args.method_invariance)
    if args.method_invariance == 0: # no regularization
        logdir = logdir + '/'
    elif args.method_invariance == 1: # data augmentation
        logdir = logdir + '_da' + str(args.data_aug_prob) + '/'
    elif args.method_invariance in [10, 100]: # data augmentation (initial implementation)
        logdir = logdir + '_da' + str(args.data_aug_prob) + '_lda' + str(args.lambda_dataaug) + '/'
    elif args.method_invariance in [2, 3, 20, 30]: # consistency regularization per layer
        if args.consis_loss != 1:
            logdir = logdir + '_da' + str(args.data_aug_prob) + '_lcon' + str(args.lambda_consis) + '_a' + str(args.alpha_layer) + '_l' + str(args.consis_loss) + '/'
        else:
            logdir = logdir + '_da' + str(args.data_aug_prob) + '_lcon' + str(args.lambda_consis) + '_a' + str(args.alpha_layer) + '/'
    elif args.method_invariance in [200, 300]: # consistency regularization per layer
        if args.consis_loss != 1:
            logdir = logdir + '_da' + str(args.data_aug_prob) + '_lda' + str(args.lambda_dataaug) + '_lcon' + str(args.lambda_consis) + '_a' + str(args.alpha_layer) + '_l' + str(args.consis_loss) + '/'
        else:
            logdir = logdir + '_da' + str(args.data_aug_prob) + '_lda' + str(args.lambda_dataaug) + '_lcon' + str(args.lambda_consis) + '_a' + str(args.alpha_layer) + '/'

    # run number
    logdir = logdir + 'run' + str(args.run_number) + '/'
    
    return logdir

# ======================================================
# ======================================================
def get_best_modelpath(models_path, prefix):

    models = os.listdir(models_path)
    model_iters = []

    for model in models:
        if prefix in model:
            a = model.find('iter')
            b = model.find('.')
            model_iters.append(int(model[a+4:b]))

    model_iters = np.array(model_iters)
    latest_iter = np.max(model_iters)

    for model in models:
        if prefix in model and str(latest_iter) in model:
            best_model = model

    return models_path + best_model

# ======================================================
# ONLY WORKS FOR BINARY SEGMENTATIONS
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

# ======================================================
# WORKS FOR BINARY AS WELL AS MULTILABEL SEGMENTATIONS
# ======================================================
def dice_score(seg1, seg2, num_labels):
    """
    Compute the Dice score between two multilabel segmentations for each label separately.
    
    Parameters
    ----------
    seg1, seg2 : numpy arrays of the same shape
        Two multilabel segmentations to compare.
    num_labels : int
        The total number of labels in the segmentations.
    
    Returns
    -------
    dice : numpy array of shape (num_labels,)
        The Dice score between the segmentations for each label separately.
    """
    dice = np.zeros(num_labels)
    
    for label in range(num_labels):
        # Create one-hot encoded segmentations for the current label
        seg1_onehot = (seg1 == label).astype(np.float32)
        seg2_onehot = (seg2 == label).astype(np.float32)

        # Compute the intersection between the segmentations
        intersection = np.sum(seg1_onehot * seg2_onehot)

        # Compute the sum of the segmentations
        seg1_sum = np.sum(seg1_onehot)
        seg2_sum = np.sum(seg2_onehot)

        # Compute the Dice score for the current label
        dice[label] = (2 * intersection) / (seg1_sum + seg2_sum)
    
    return dice[1:] # ignore background dice
