import data_placenta_new as data_placenta
import data_prostate
import argparse
import logging
import numpy as np

# ==========================================
# ==========================================
def load_data(args,
              sub_dataset,
              train_test_val):

    if args.dataset == 'placenta':
        data_orig_path = '/data/vision/polina/projects/fetal/projects/placenta-segmentation/data/split-nifti-processed/' # orig data is here
        data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/placenta/' # save processed data here
        data = data_placenta.load_dataset(data_orig_path,
                                          data_proc_path,
                                          train_test_val = train_test_val,
                                          cv_fold = args.cv_fold_num)

    elif args.dataset == 'prostate':
        data_orig_path = '/data/vision/polina/users/nkarani/data/segmentation/prostate/' # orig data is here
        data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/prostate/' # save processed data here
        # data_proc_path = '/data/scratch/nkarani/projects/crael/seg/data/prostate/' # save processed data here
        data = data_prostate.load_dataset(data_orig_path,
                                          data_proc_path,
                                          sub_dataset,
                                          train_test_val = train_test_val,
                                          cv_fold = args.cv_fold_num)

    return data

# ==========================================
# ==========================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'test data loader')
    parser.add_argument('--dataset', default='prostate') # placenta / prostate
    parser.add_argument('--sub_dataset', default='RUNMC') # prostate: BIDMC / BMC / HK / I2CVB / RUNMC / UCL
    parser.add_argument('--cv_fold_num', default=2, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    logging.info('Reading training and validation data')
    data_tr = load_data(args, args.sub_dataset, 'train')
    data_vl = load_data(args, args.sub_dataset, 'validation')

    images_tr = data_tr["images"]
    labels_tr = data_tr["labels"]
    subject_names_tr = data_tr["subject_names"]

    images_vl = data_vl["images"]
    labels_vl = data_vl["labels"]
    subject_names_vl = data_vl["subject_names"]

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