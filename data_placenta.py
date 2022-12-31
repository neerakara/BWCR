# ==================================================
# import stuff
# ==================================================
import numpy as np
import os
import utils_data
import logging
import gc
import h5py

import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.use('agg')
import matplotlib.pyplot as plt 

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
# Test set is the same for all settings -> 50 images
# Val set is the same for all settings -> 5 images
# ==================================================
def get_train_test_val_split_ids(cv_fold):

    train_test_val_split_ids = {}

    if cv_fold == 1: # 'small training dataset'
        train_test_val_split_ids['test'] = np.arange(0, 50, 1).tolist()
        train_test_val_split_ids['train'] = np.arange(50, 55, 1).tolist()
        train_test_val_split_ids['validation'] = np.arange(80, 85, 1).tolist()

    elif cv_fold == 2: # 'medium training dataset'
        train_test_val_split_ids['test'] = np.arange(0, 50, 1).tolist()
        train_test_val_split_ids['train'] = np.arange(50, 65, 1).tolist()
        train_test_val_split_ids['validation'] = np.arange(80, 85, 1).tolist()

    elif cv_fold == 3: # 'large training dataset'
        train_test_val_split_ids['test'] = np.arange(0, 50, 1).tolist()
        train_test_val_split_ids['train'] = np.arange(50, 75, 1).tolist()
        train_test_val_split_ids['validation'] = np.arange(80, 85, 1).tolist()

    return train_test_val_split_ids

# ==================================================
# count number of 2d slices in the 3d images of the reqeusted subject IDs
# ==================================================
def count_total_slices(image_paths, sub_ids):
    
    num_slices = 0
    for n in range(len(sub_ids)):
        image_path = image_paths[sub_ids[n]]
        image = utils_data.load_nii(image_path)[0].astype(float)
        num_slices = num_slices + image.shape[2]

    return num_slices

# ==================================================
# Loads the image / label from the source file (e.g. nii) as a numpy array
# ==================================================
def prepare_dataset(data_orig_path,
                    output_file,
                    train_test_val = 'train',
                    cv_fold = 1,
                    size = (128, 128),
                    target_res = (1.0, 1.0)): # orig images are 1x1x1 isotropic resolution

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
    # count number of slices to pre-define dataset size
    # ===============================
    if DEBUGGING == 1: logging.info('Counting dataset size...')
    num_slices = count_total_slices(image_paths_all, sub_ids)

    # ===============================
    # Create datasets for images and labels
    # ===============================
    images_size = list(size) + [num_slices]
    data = {}
    data['images'] = hdf5_file.create_dataset("images", images_size, dtype=np.float32)
    data['labels'] = hdf5_file.create_dataset("labels", images_size, dtype=np.uint8)

    # ===============================
    # initialize lists
    # ===============================
    image_paths = []
    label_paths = []
    depths = []
    subject_names = []

    # ===============================
    # helper counter for writing data to file
    # ===============================
    counter_from = 0

    # ===============================
    # read each subject's data, pre-process it and add it to the hdf5
    # ===============================
    for n in range(len(sub_ids)):
    
        # ==================
        # read image
        # ==================
        sub_name = sub_names_all[sub_ids[n]]
        image_path = image_paths_all[sub_ids[n]]
        image, aff, hdr = utils_data.load_nii(image_path)
        image = image.astype(float)
        if DEBUGGING == 1:
            print('image stats before norm (min, max, mean): ' + str(np.min(image)) + ', ' + str(np.max(image)) + ', ' + str(np.mean(image)))
    
        # ==================
        # normalize image intensities
        # ==================
        image = utils_data.normalize_intensities(image)
        if DEBUGGING == 1:
            print('image stats after norm (min, max, mean): ' + str(np.min(image)) + ', ' + str(np.max(image)) + ', ' + str(np.mean(image)))

        # ==================
        # read label
        # ==================
        label_path = label_paths_all[sub_ids[n]]
        label, aff, hdr = utils_data.load_nii(label_path)
        label[label != 0.0] = 1.0
        if DEBUGGING == 1:
            print('number of unique labels: ' + str(np.unique(label)))

        if DEBUGGING == 1:
            print(image_path)
            print(image.shape)
            print(label_path)
            print(label.shape)

        # ==================
        # squeeze
        # ==================
        image = np.squeeze(image)
        label = np.squeeze(label)

        # ==================
        # crop or pad to make the in-plane image size the same for all subjects
        # ==================
        image = utils_data.crop_or_pad_volume_in_xy(image, size[0], size[1])
        label = utils_data.crop_or_pad_volume_in_xy(label, size[0], size[1])

        # ==================
        # make in-plane resolution the same for all subjects
        # ==================
        # not doing this for placenta, as this is already the case in the raw images

        # ==================
        # write image and label to hdf5 file
        # ==================
        _write_range_to_hdf5(data,
                             image,
                             label,
                             counter_from,
                             counter_from + image.shape[-1])
        
        counter_from = counter_from + image.shape[-1]

        # ==================
        # append remaining attributes to respective lists
        # ==================
        image_paths.append(image_path)
        label_paths.append(label_path)
        depths.append(image.shape[-1])
        subject_names.append(sub_name)

    # ==================
    # Write the small datasets
    # ==================
    hdf5_file.create_dataset('depths', data=np.asarray(depths, dtype=np.uint16))
    hdf5_file.create_dataset('subject_names', data=np.asarray(subject_names, dtype="S10"))
    
    # ==================
    # After test train loop
    # ==================
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
                 size,
                 target_resolution,
                 force_overwrite=False):

    size_str = '_size_' + '_'.join([str(i) for i in size])
    res_str = '_res_' + '_'.join([str(i) for i in target_resolution])
    data_filename = 'cv' + str(cv_fold) + '_' + str(train_test_val) + size_str + res_str + '.hdf5'
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

# ==========================================================
# ==========================================================
def normalize_img_for_vis(img):

    if np.percentile(img, 99) == np.percentile(img, 1):
        epsilon = 0.0001
    else:
        epsilon = 0.0
    img = (img - np.percentile(img, 1)) / (np.percentile(img, 99) - np.percentile(img, 1) + epsilon)
    img[img<0] = 0.0
    img[img>1] = 1.0

    return (img * 255).astype(np.uint8)

# ==========================================
# ==========================================
if __name__ == "__main__":

    # setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    cv_fold = 3 # 1 / 2 / 3
    train_test_val = 'test' # train / validation / test
    size = (128, 128)
    target_resolution = (1.0, 1.0)
    data_orig_path = '/data/vision/polina/projects/fetal/projects/placenta-segmentation/data/split-nifti-raw/' # orig data is here
    data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/placenta/' # save processed data here
    data = load_dataset(data_orig_path,
                        data_proc_path,
                        train_test_val = train_test_val,
                        cv_fold = cv_fold,
                        size = size,
                        target_resolution = target_resolution)

    # visualize
    images = data['images']
    labels = data['labels']
    subnames = data['subject_names']
    depths = data['depths']
    logging.info(images.shape)
    logging.info(labels.shape)
    logging.info(subnames)
    logging.info(depths)

    plt.figure(figsize=(24, 6))
    k=0
    num_subjects = len(subnames)
    for s in range(8):
        if num_subjects < s + 1:
            continue
        
        # get subject data
        logging.info(subnames[s])
        sub_start = int(np.sum(depths[:s]))
        sub_end = int(np.sum(depths[:s+1]))
        subject_image = images[:,:,sub_start:sub_end]
        subject_label = labels[:,:,sub_start:sub_end]
        
        # find slice with largest placenta
        logging.info(subject_label.shape)
        placenta_sizes = np.sum(subject_label, axis=(0,1))
        logging.info(placenta_sizes.shape)
        idx_largest = np.argmax(placenta_sizes)
        logging.info(idx_largest)
        slice_image = subject_image[:, :, idx_largest]
        slice_label = subject_label[:, :, idx_largest]

        # plot
        plt.subplot(2, 8, s + 1, xticks=[], yticks=[])
        plt.imshow(np.rot90(normalize_img_for_vis(slice_image),k), cmap = 'gray')
        plt.subplot(2, 8, s + 9, xticks=[], yticks=[])
        plt.imshow(np.rot90(normalize_img_for_vis(slice_label),k), cmap = 'gray')

    savepath = data_proc_path + 'cv_' + str(cv_fold) + '_' + train_test_val + '.png'
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()