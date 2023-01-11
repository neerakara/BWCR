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
# ==================================================
def get_image_and_label_paths(data_path,
                              sub_dataset,
                              train_test_val):

    if train_test_val in ['train', 'validation']:
        data_path = data_path + train_test_val + '/'
    else:
        if sub_dataset == 'InD':
            data_path = data_path + train_test_val + '_in/'
        elif sub_dataset == 'OoD':
            data_path = data_path + train_test_val + '_out/'
    
    image_paths = []
    brainmask_paths = []
    label_paths = []
    sub_names = []
    for filename in os.listdir(data_path):
        if 'FLAIR' in filename:
            sub_name = filename[:filename.find('_')]
            sub_names.append(sub_name)
            image_paths.append(data_path + str(sub_name) + '_FLAIR_isovox.nii.gz')
            brainmask_paths.append(data_path + str(sub_name) + '_isovox_fg_mask.nii.gz')
            label_paths.append(data_path + str(sub_name) + '_gt_isovox.nii.gz')

    return sub_names, image_paths, brainmask_paths, label_paths

# ==================================================
# count number of 2d slices in the 3d images of the reqeusted subject IDs
# ==================================================
def count_total_slices(image_paths):
    
    num_slices = 0
    for image_path in image_paths:
        image = utils_data.load_nii(image_path)[0].astype(float)
        num_zz = np.sum(image, axis=(0,1))
        num_slices = num_slices + len(np.where(num_zz != 0)[0])

    return num_slices

# ==================================================
# Loads the image / label from the source file (e.g. nii) as a numpy array
# ==================================================
def prepare_dataset(data_orig_path,
                    output_file,
                    sub_dataset,
                    train_test_val = 'train',
                    size = (256, 256),
                    target_res = (1.0, 1.0)): # orig images are 1x1x1 isotropic resolution

    # =======================
    # create a hdf5 file to store all requested data
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # get paths of all images and labels for this subdataset
    if DEBUGGING == 1: logging.info('Reading image and label paths...')
    sub_names_all, image_paths_all, brain_mask_paths_all, label_paths_all = get_image_and_label_paths(data_orig_path,
                                                                                                      sub_dataset,
                                                                                                      train_test_val)
    
    # ===============================
    # count number of slices to pre-define dataset size
    # ===============================
    if DEBUGGING == 1: logging.info('Counting dataset size...')
    num_slices = count_total_slices(brain_mask_paths_all)

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
    for n in range(len(sub_names_all)):
    
        # ==================
        # read image
        # ==================
        sub_name = sub_names_all[n]
        image_path = image_paths_all[n]
        image, aff, hdr = utils_data.load_nii(image_path)
        image = image.astype(float)
        if DEBUGGING == 1:
            print('image stats before norm (min, max, mean): ' + str(np.min(image)) + ', ' + str(np.max(image)) + ', ' + str(np.mean(image)))
    
        # ==================
        # read the brain mask
        # ==================
        brainmask, _, _ = utils_data.load_nii(brain_mask_paths_all[n])          

        # ==================
        # remove slices with all zeros in the axial direction
        # ==================
        nonzero_slices = np.where(np.sum(brainmask, axis=(0,1)) != 0)[0]
        image = image[:, :, nonzero_slices]

        # ==================
        # normalize image intensities
        # ==================
        image = utils_data.normalize_intensities_flair(image)

        if DEBUGGING == 1:
            print('image stats after norm (min, max, mean): ' + str(np.min(image)) + ', ' + str(np.max(image)) + ', ' + str(np.mean(image)))

        # ==================
        # read label
        # ==================
        label_path = label_paths_all[n]
        label, aff, hdr = utils_data.load_nii(label_path)
        label[label != 0.0] = 1.0
        if DEBUGGING == 1:
            print('number of unique labels: ' + str(np.unique(label)))
        logging.info(label.shape)
        label = label[:, :, nonzero_slices]
        logging.info(label.shape)

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
        # for some subjects (from BEST), the orientation of axial slices is different than the rest
        # ==================
        logging.info(sub_name)
        if sub_dataset == 'InD':
            cond1 = (train_test_val == 'validation' and sub_name in ['6', '7'])
            cond2 = (train_test_val == 'train' and sub_name in ['24', '25', '26', '27', '28', '29', '30', '31', '32', '33'])
            cond3 = (train_test_val == 'test' and sub_name in ['25', '26', '27', '28', '29', '30', '31', '32', '33'])
            if cond1 or cond2 or cond3:
                for zz in range(image.shape[-1]):
                    image[:, :, zz] = np.rot90(image[:, :, zz], 2)
                    label[:, :, zz] = np.rot90(label[:, :, zz], 2)

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
                 sub_dataset,
                 train_test_val,
                 size,
                 target_resolution,
                 force_overwrite=False):

    size_str = '_size_' + '_'.join([str(i) for i in size])
    res_str = '_res_' + '_'.join([str(i) for i in target_resolution])
    data_filename = sub_dataset + '_' + str(train_test_val) + size_str + res_str + '.hdf5'
    data_filepath = preprocessing_folder + data_filename

    if not os.path.exists(data_filepath) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_dataset(input_folder,
                        data_filepath,
                        sub_dataset,
                        train_test_val,
                        size,
                        target_resolution)
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

    train_test_val = 'test' # train / validation / test
    sub_dataset = 'OoD' # InD / OoD
    size = (192, 192)
    target_resolution = (1.0, 1.0)
    data_orig_path = '/data/vision/polina/users/nkarani/data/segmentation/ms_lesions/shifts_ms/' # orig data is here
    data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/ms/' # save processed data here
    data = load_dataset(data_orig_path,
                        data_proc_path,
                        sub_dataset,
                        train_test_val = train_test_val,
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

    k=-1
    num_subjects = len(subnames)
    n = num_subjects
    plt.figure(figsize=(num_subjects*4, 18))
    for s in range(n):
        if num_subjects < s + 1:
            continue
        
        # get subject data
        logging.info(subnames[s])
        sub_start = int(np.sum(depths[:s]))
        sub_end = int(np.sum(depths[:s+1]))
        subject_image = images[:,:,sub_start:sub_end]
        subject_label = labels[:,:,sub_start:sub_end]

        # plot histogram of subject's image
        histogram, bin_edges = np.histogram(subject_image, bins=512)
        nwm = np.argsort(histogram)[-2]
        
        # find slice with largest fg
        logging.info(subject_label.shape)
        fg_sizes = np.sum(subject_label, axis=(0,1))
        logging.info(fg_sizes.shape)
        idx_largest = np.argmax(fg_sizes)
        logging.info(idx_largest)
        slice_image = subject_image[:, :, idx_largest]
        slice_label = subject_label[:, :, idx_largest]

        # plot
        plt.subplot(3, n, s + 1, xticks=[], yticks=[])
        plt.imshow(np.rot90(slice_image,k), cmap = 'gray')
        plt.colorbar()
        plt.title(subnames[s])
        plt.subplot(3, n, s + n + 1, xticks=[], yticks=[])
        plt.imshow(np.rot90(slice_label,k), cmap = 'gray')
        plt.colorbar()
        plt.subplot(3, n, s + 2*n + 1)
        plt.plot(bin_edges[0:-1], histogram) 
        plt.axvline(x = bin_edges[nwm], color = 'r')
        plt.ylim([0.0, 50000.0])
        

    savepath = data_proc_path + train_test_val + '_' + sub_dataset + '.png'
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()