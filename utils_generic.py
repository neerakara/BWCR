# ======================================================
# ======================================================
def make_expdir(args):
    logdir = args.save_path + args.dataset + '_cv' + str(args.cv_fold_num) + '/' # data related
    logdir = logdir + 'lr' + str(args.lr) + '_bsize' + str(args.batch_size) + '/' # optimization related
    logdir = logdir + 'method_invariance_' + str(args.method_invariance) # invariance method
    if args.method_invariance == 0: # no regularization
        logdir = logdir + '/'
    elif args.method_invariance == 1: # data augmentation
        logdir = logdir + '_da_prob' + str(args.data_aug_prob) + '_lam' + str(args.lambda_reg) + '/'
    elif args.method_invariance == 2: # consistency regularization
        logdir = logdir + '_da_prob' + str(args.data_aug_prob) + '_lam' + str(args.lambda_reg) + '/'
    elif args.method_invariance == 3: # consistency regularization per layer
        logdir = logdir + '_da_prob' + str(args.data_aug_prob) + '_lam' + str(args.lambda_reg) + '_alp' + str(args.alpha_layer) + '/'
    logdir = logdir + 'run' + str(args.run_number) + '/'
    return logdir