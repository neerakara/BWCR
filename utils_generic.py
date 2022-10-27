# ======================================================
# ======================================================
def make_expdir(args):
    logdir = args.save_path + args.dataset + '_cv' + str(args.cv_fold_num) + '/' # data related
    logdir = logdir + 'lr' + str(args.lr) + '_bsize' + str(args.batch_size) + '/' # optimization related    
    logdir = logdir + '_data_aug_' + str(args.data_aug_prob) + '/' # data augmentation related
    logdir = logdir + 'run' + str(args.run_number) + '/'
    return logdir     