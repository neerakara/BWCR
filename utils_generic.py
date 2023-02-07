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

    # optimization related
    logdir = logdir + 'lr' + str(args.lr) + '_bs' + str(args.batch_size) + '/' 

    # invariance method
    logdir = logdir + 'l0_' + str(args.l0) 
    logdir = logdir + '_l1_' + args.l1_loss + '_' + str(args.l1) 
    logdir = logdir + '_l2_' + args.l2_loss + '_' + str(args.l2) 
    if args.l2_loss == 'l2_margin':
        logdir = logdir + '_' + str(args.l2_loss_margin)
    if args.teacher != 'self':
        logdir = logdir + '_teacher_' + args.teacher
    logdir = logdir + '_t_' + str(args.temp) + '/'

    # run number
    logdir = logdir + 'run' + str(args.run_number) + '/'
    
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