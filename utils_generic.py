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
        logdir = logdir + '_da_prob' + str(args.data_aug_prob) + '_lam' + str(args.lambda_data_aug) + '/'
    elif args.method_invariance == 2: # consistency regularization
        logdir = logdir + '_da_prob' + str(args.data_aug_prob) + '_lam' + str(args.lambda_consis) + '/'
    elif args.method_invariance == 3: # consistency regularization per layer
        logdir = logdir + '_da_prob' + str(args.data_aug_prob) + '_lam_da' + str(args.lambda_data_aug) + '_lam_con' + str(args.lambda_consis) + '_alp' + str(args.alpha_layer) + '/'

    # run number
    logdir = logdir + 'run' + str(args.run_number) + '/'
    
    return logdir