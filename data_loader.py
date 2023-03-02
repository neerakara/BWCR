import data_placenta
import data_nci
import data_prostate_nci
import data_prostate_promise
import data_ms
import data_acdc
import argparse
import logging
import numpy as np

# ==========================================
# ==========================================
def load_data(dataset,
              sub_dataset,
              cv_fold_num,
              train_test_val):

    if dataset == 'placenta':
        data_orig_path = '/data/vision/polina/projects/fetal/projects/placenta-segmentation/data/split-nifti-raw/' # orig data is here
        data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/placenta/' # save processed data here
        data = data_placenta.load_dataset(data_orig_path,
                                          data_proc_path,
                                          train_test_val = train_test_val,
                                          cv_fold = cv_fold_num,
                                          size = (128, 128),
                                          target_resolution = (1.0, 1.0))

    elif dataset == 'prostate':
        data_orig_path = '/data/vision/polina/users/nkarani/data/segmentation/prostate/original/' # orig data is here
        data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/prostate/' # save processed data here
        # data_proc_path = '/data/scratch/nkarani/projects/crael/seg/data/prostate/' # save processed data here
        if sub_dataset in ['RUNMC', 'BMC']:
            data = data_prostate_nci.load_dataset(data_orig_path + 'nci/',
                                                  data_proc_path,
                                                  sub_dataset,
                                                  train_test_val = train_test_val,
                                                  cv_fold = cv_fold_num,
                                                  size = (256, 256),
                                                  target_resolution = (0.625, 0.625))
            
        elif sub_dataset in ['nci']:
            data = data_nci.load_dataset(data_orig_path + 'nci/',
                                         data_proc_path,
                                         train_test_val = train_test_val,
                                         cv_fold = cv_fold_num,
                                         size = (192, 192),
                                         target_resolution = (0.625, 0.625))

        elif sub_dataset in ['UCL', 'HK', 'BIDMC']:
            data = data_prostate_promise.load_dataset(data_orig_path + 'promise/',
                                                      data_proc_path,
                                                      sub_dataset,
                                                      train_test_val = train_test_val,
                                                      cv_fold = cv_fold_num,
                                                      size = (256, 256),
                                                      target_resolution = (0.625, 0.625))

    elif dataset == 'ms':
        data_orig_path = '/data/vision/polina/users/nkarani/data/segmentation/ms_lesions/shifts_ms/' # orig data is here
        data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/ms/' # save processed data here
        data = data_ms.load_dataset(data_orig_path,
                                    data_proc_path,
                                    sub_dataset, # InD / OoD
                                    train_test_val = train_test_val,
                                    size = (192, 192),
                                    target_resolution = (1.0, 1.0))
        
    elif dataset == 'acdc':
        data_orig_path = '/data/vision/polina/users/nkarani/data/segmentation/acdc/' # orig data is here
        data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/acdc/' # save processed data here
        data = data_acdc.load_dataset(data_orig_path,
                                      data_proc_path,
                                      train_test_val = train_test_val,
                                      cv_fold = cv_fold_num,
                                      size = (192, 192),
                                      target_resolution = (1.33, 1.33))

    return data

# ==========================================
# ==========================================
def load_without_preproc(dataset,
                         sub_dataset,
                         subject_name,
                         ttv):

    if dataset == 'prostate':
        data_orig_path = '/data/vision/polina/users/nkarani/data/segmentation/prostate/original/' # orig data is here
        data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/prostate/' # save processed data here
        # data_proc_path = '/data/scratch/nkarani/projects/crael/seg/data/prostate/' # save processed data here
        if sub_dataset in ['RUNMC', 'BMC', 'nci']:
            image, label = data_prostate_nci.load_without_preproc(data_proc_path,
                                                                  subject_name)

        elif sub_dataset in ['UCL', 'HK', 'BIDMC']:
            image, label = data_prostate_promise.load_without_preproc(data_proc_path,
                                                                      subject_name)
            
    elif dataset == 'acdc':
        if ttv == 'test':
            data_orig_path = '/data/vision/polina/users/nkarani/data/segmentation/acdc/testing/' # orig data is here
        else:
            data_orig_path = '/data/vision/polina/users/nkarani/data/segmentation/acdc/training/'

        image, label = data_acdc.load_without_preproc(data_orig_path,
                                                      subject_name)
        

    return image, label

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
    data_tr = load_data(args.dataset, args.sub_dataset, args.cv_fold_num, 'train')
    data_vl = load_data(args.dataset, args.sub_dataset, args.cv_fold_num, 'validation')

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