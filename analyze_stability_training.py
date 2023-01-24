import numpy as np
import logging
import utils_vis

# ==========================================
# ==========================================
if __name__ == "__main__":

    # ===================================
    # setup logging
    # ===================================
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # ===================================
    # paths
    # ===================================
    dataset = 'prostate'
    cvs = [2]
    runs = [1, 2]
    # suffixes = ['m0', 'm100_da0.5_lda1.0', 'm200_da0.5_lda1.0_lcon0.01_a100.0_l2', 'm200_da0.5_lda1.0_lcon0.01_a10.0_l2', 'm200_da0.5_lda1.0_lcon0.01_a1.0_l2', 'm200_da0.5_lda1.0_lcon0.01_a0.1_l2']
    suffixes = ['m0', 'm100_da0.5_lda1.0', 'm200_da0.5_lda1.0_lcon0.01_a100.0_l2']
    # suffixes = ['m100_da0.5_lda1.0']

    for cv in cvs:
        for r in runs:
            if dataset == 'prostate':
                basepath = '/data/scratch/nkarani/projects/crael/seg/logdir/v6/prostate_RUNMC_cv' + str(cv) + '/lr0.0001_sch2_bs16/unet/'
            elif dataset == 'ms':
                basepath = '/data/scratch/nkarani/projects/crael/seg/logdir/v6/ms/lr0.0001_sch2_bs16/unet/'
            res = []
            leg = []
            for s in suffixes:
                result = np.load(basepath + s + '/run' + str(r) + '/models/variance_across_training_iters.npy')
                logging.info(result.shape)
                utils_vis.plot_scatter(result[-499:,10:], basepath + s + '/run' + str(r) + '/models/variance_across_training_iters.png')
                utils_vis.plot_subjectwise(result[-499:,10:], basepath + s + '/run' + str(r) + '/models/variance_across_training_iters_subjectwise.png')

                res.append(np.mean(result[-499:, 10:], 1))
                leg.append(s)

            utils_vis.plot_methods(res, leg, basepath + 'average_results_over_subjects_for_500_models_run' + str(r) + '.png')
