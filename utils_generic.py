import os
import numpy as np

# ======================================================
# ======================================================
def make_expdir(args):

    # data related
    if args.dataset == 'placenta':
        logdir = args.save_path + args.dataset + '_cv' + str(args.cv_fold_num) + '/' 
    elif args.dataset == 'prostate':
        logdir = args.save_path + args.dataset + '_' + args.sub_dataset + '_cv' + str(args.cv_fold_num) + '/'

    # optimization related
    logdir = logdir + 'lr' + str(args.lr) + '_bsize' + str(args.batch_size) + '/' 

    # model related
    if args.model_has_heads == 1:
        logdir = logdir + 'unet_with_heads/'
    else:
        logdir = logdir + 'unet/'

    # invariance method
    logdir = logdir + 'method_invariance_' + str(args.method_invariance)
    if args.method_invariance == 0: # no regularization
        logdir = logdir + '/'
    elif args.method_invariance == 1: # data augmentation
        logdir = logdir + '_da_prob' + str(args.data_aug_prob) + '/'
    elif args.method_invariance in [10, 100]: # data augmentation (initial implementation)
        logdir = logdir + '_da_prob' + str(args.data_aug_prob) + '_lam_da' + str(args.lambda_dataaug) + '/'
    elif args.method_invariance in [2, 3, 20, 30]: # consistency regularization per layer
        logdir = logdir + '_da_prob' + str(args.data_aug_prob) + '_lam_con' + str(args.lambda_consis) + '_alp' + str(args.alpha_layer) + '/'
    elif args.method_invariance in [200, 300]: # consistency regularization per layer
        logdir = logdir + '_da_prob' + str(args.data_aug_prob) + '_lam_da' + str(args.lambda_dataaug) + '_lam_con' + str(args.lambda_consis) + '_alp' + str(args.alpha_layer) + '/'

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

    