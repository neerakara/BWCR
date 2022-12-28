# ==================================================
# import stuff
# ==================================================
import nibabel as nib
import numpy as np
import os
import utils_data
import logging
import gc
import h5py

DEBUGGING = 1

# ==================================================
# get image and label paths
# for now, only considering subjects containing only one time point labelled in the time series
# ==================================================
def get_image_and_label_paths(data_path):

    subdirs = []
    for dir in os.listdir(data_path):
        if os.path.isdir(data_path + dir) and 'MAP' in dir:
            subdirs.append(data_path + dir)

    image_paths = []
    label_paths = []
    sub_names = []
    for subdir in subdirs:
        sub_name = subdir[subdir.rfind('/'):][:9]
        print(sub_name)
        if sub_name not in sub_names:
            sub_names.append(sub_name)
            image_dir = subdir + '/volume/'
            label_dir = subdir + '/segmentation/'
            image_paths.append(image_dir + os.listdir(image_dir)[0])
            label_paths.append(label_dir + os.listdir(label_dir)[0])

    return sub_names, image_paths, label_paths

# ==================================================
# 3 settings: small (5), medium (15) and large (25) training dataset
# ==================================================
def get_train_test_val_split_ids(cv_fold):

    train_test_val_split_ids = {}

    if cv_fold == 1: # 'small training dataset'
        train_test_val_split_ids['test'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        train_test_val_split_ids['train'] = [50, 51, 52, 53, 54]
        train_test_val_split_ids['validation'] = [80, 81, 82, 83, 84]

    elif cv_fold == 2: # 'medium training dataset'
        train_test_val_split_ids['test'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_test_val_split_ids['train'] = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
        train_test_val_split_ids['validation'] = [80, 81, 82, 83, 84]

    elif cv_fold == 3: # 'large training dataset'
        train_test_val_split_ids['test'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_test_val_split_ids['train'] = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
        train_test_val_split_ids['validation'] = [80, 81, 82, 83, 84]

    return train_test_val_split_ids

# ==================================================
# ==================================================
def count_total_size(image_paths, sub_ids):
    
    num_slices = 0

    for n in range(len(sub_ids)):
    
        # read image and count number of 2d slices in this 3d image
        image_path = image_paths[sub_ids[n]]
        image = nib.load(image_path)
        image = np.array(image.dataobj)
        image = image.astype(float)
        num_slices = num_slices + image.shape[2]

    images_size = [image.shape[0], image.shape[1], num_slices] # assuming all images have same in-plane dimensions

    return images_size

# ==================================================
# Loads the image / label from the source file (e.g. nii) as a numpy array
# ==================================================
def prepare_dataset(data_orig_path,
                    output_file,
                    train_test_val = 'train',
                    cv_fold = 1):

    # =======================
    # create a hdf5 file to store all requested data
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # get paths of all images and labels for this subdataset
    if DEBUGGING == 1: logging.info('Reading image and label paths...')
    sub_names_all, image_paths_all, label_paths_all = get_image_and_label_paths(data_orig_path)

    # get ids for this train / test / validation split
    if DEBUGGING == 1: logging.info('Getting ids of subjects to be read...')
    train_test_val_split_ids = get_train_test_val_split_ids(cv_fold)
    sub_ids = train_test_val_split_ids[train_test_val]
    
    # ===============================
    # Create datasets for images and labels
    # ===============================
    data = {}

    # count number of slices to pre-define dataset size
    if DEBUGGING == 1: logging.info('Counting dataset size...')
    images_size = count_total_size(image_paths_all,
                                   sub_ids)

    data['images'] = hdf5_file.create_dataset("images", images_size, dtype=np.float32)
    data['labels'] = hdf5_file.create_dataset("labels", images_size, dtype=np.uint8)

    # initialize lists
    image_paths = []
    label_paths = []
    depths = []
    subject_names = []

    # helper counter for writing data to file
    counter_from = 0

    for n in range(len(sub_ids)):
    
        # read image
        image_path = image_paths_all[sub_ids[n]]
        image = nib.load(image_path)
        image = np.array(image.dataobj)
        image = image.astype(float)
        if DEBUGGING == 1:
            print('image stats before norm (min, max, mean): ' + str(np.min(image)) + ', ' + str(np.max(image)) + ', ' + str(np.mean(image)))
    
        # normalize image intensities
        image = utils_data.normalize_intensities(image)
        if DEBUGGING == 1:
            print('image stats after norm (min, max, mean): ' + str(np.min(image)) + ', ' + str(np.max(image)) + ', ' + str(np.mean(image)))

        # read label
        label_path = label_paths_all[sub_ids[n]]
        label = nib.load(label_path)
        label = np.array(label.dataobj)
        label[label != 0.0] = 1.0
        if DEBUGGING == 1:
            print('number of unique labels: ' + str(np.unique(label)))

        if DEBUGGING == 1:
            print(image_path)
            print(image.shape)
            print(label_path)
            print(label.shape)

        # squeeze
        image = np.squeeze(image)
        label = np.squeeze(label)

        # write image and label to hdf5 file
        _write_range_to_hdf5(data,
                             image,
                             label,
                             counter_from,
                             counter_from + image.shape[-1])
        
        counter_from = counter_from + image.shape[-1]

        # append remaining attributes to respective lists
        image_paths.append(image_path)
        label_paths.append(label_path)
        depths.append(image.shape[-1])
        subject_names.append(image_path[-13:-7])
    
    # Write the small datasets
    hdf5_file.create_dataset('depths', data=np.asarray(depths, dtype=np.uint16))
    hdf5_file.create_dataset('subject_names', data=np.asarray(subject_names, dtype="S10"))
    
    # After test train loop:
    hdf5_file.close()
    
    return 0

# ===============================================================
# Helper function to write a range of data to the hdf5 datasets
# ===============================================================
def _write_range_to_hdf5(hdf5_data,
                         images,
                         labels,
                         counter_from,
                         counter_to):

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    hdf5_data['images'][..., counter_from : counter_to] = images
    hdf5_data['labels'][..., counter_from : counter_to] = labels

    return 0

# ===============================================================
# ===============================================================
def load_dataset(input_folder,
                 preprocessing_folder,
                 train_test_val,
                 cv_fold,
                 force_overwrite=False):

    data_filename = 'cv' + str(cv_fold) + '_' + str(train_test_val) + '.hdf5'
    data_filepath = preprocessing_folder + data_filename

    if not os.path.exists(data_filepath) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_dataset(input_folder,
                        data_filepath,
                        train_test_val,
                        cv_fold)
    else:
        logging.info('Already preprocessed. Loading now!')

    return h5py.File(data_filepath, 'r')

# ==========================================
# ==========================================
if __name__ == "__main__":
    cv_fold = 1
    train_test_val = 'train'
    data_orig_path = '/data/vision/polina/projects/fetal/projects/placenta-segmentation/data/split-nifti-processed/' # orig data is here
    data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/placenta/' # save processed data here
    data = load_dataset(data_orig_path,
                        data_proc_path,
                        train_test_val = train_test_val,
                        cv_fold = cv_fold)