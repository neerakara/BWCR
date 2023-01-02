import os
import numpy as np
import logging
import gc
import h5py
from skimage import transform
import utils_data
import pydicom as dicom
import nrrd
import re
import skimage.io as io
import SimpleITK as sitk
import subprocess
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.use('agg')
import matplotlib.pyplot as plt 
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# IDs of the different sub-datasets within the PROMISE12 dataset
RUNMC_IDS = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] # Overlap with the NCI dataset | Ignore these images
UCL_IDS = [1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
BIDMC_IDS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
HK_IDS = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

# ==================================================
# Test set is the same for all settings -> 10 images
# ==================================================
def get_train_test_val_split_ids(cv_fold_num):

    train_test_val_split_ids = {}
    train_test_val_split_ids['test'] = np.arange(0, 10, 1).tolist()

    return train_test_val_split_ids

# ===============================================================
# ===============================================================
def count_total_slices(image_path_list, sub_ids):

    num_slices = 0

    for n in range(len(sub_ids)):
        image_path = image_path_list[sub_ids[n]]
        image = sitk.ReadImage(image_path)
        num_slices += sitk.GetArrayFromImage(image).shape[0]

    return num_slices

# ===============================================================
# Get image paths of all subjects of this dataset
# ===============================================================
def get_image_path_list(data_orig_path,
                        data_proc_path,
                        sub_dataset):

    if sub_dataset == 'RUNMC':
        sub_dataset_ids = RUNMC_IDS
    elif sub_dataset == 'UCL':
        sub_dataset_ids = UCL_IDS
    elif sub_dataset == 'BIDMC':
        sub_dataset_ids = BIDMC_IDS
    elif sub_dataset == 'HK':
        sub_dataset_ids = HK_IDS

    image_path_list = []
    for _, _, fileList in os.walk(data_orig_path):
        for filename in fileList:
            if re.match(r'Case\d\d.mhd', filename):
                patient_id = filename[4:6]
                if int(patient_id) in sub_dataset_ids:
                    image_path_list.append(data_orig_path + filename)

    return image_path_list

# ===============================================================
# ===============================================================
def prepare_data(input_folder,
                 preprocessing_folder,
                 output_file,
                 sub_dataset, # UCL / HK / BIDMC
                 train_test_val,
                 cv_fold_num,
                 size,
                 target_resolution):

    # =======================
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # =======================
    # =======================
    logging.info('Counting files and parsing meta data...')
    image_path_list = get_image_path_list(input_folder, preprocessing_folder, sub_dataset)

    # get ids for this train / test / validation split
    logging.info('Getting ids of subjects to be read...')
    train_test_val_split_ids = get_train_test_val_split_ids(cv_fold_num)
    sub_ids = train_test_val_split_ids[train_test_val]
    
    # ===============================
    # count number of slices to pre-define dataset size
    # ===============================
    logging.info('Counting dataset size...')
    num_slices = count_total_slices(image_path_list, sub_ids)
    nx, ny = size

    # =======================
    # Create datasets for images and masks
    # =======================
    images_size = list(size) + [num_slices]
    logging.info("---------------------------------")
    logging.info("Creating HDF5 for storing images and labels of size: " + str(images_size))
    logging.info("---------------------------------")
    data = {}
    data['images'] = hdf5_file.create_dataset("images", images_size, dtype=np.float32)
    data['labels'] = hdf5_file.create_dataset("labels", images_size, dtype=np.uint8)

    # ===============================
    # initialize lists
    # ===============================
    subject_names = []
    nx_list = []
    ny_list = []
    nz_list = []
    px_list = []
    py_list = []
    pz_list = []    

    # ===============================
    # helper counter for writing data to file
    # ===============================
    counter_from = 0

    # =======================
    # read each subject's image, pre-process it and write to the hdf5 file
    # =======================
    logging.info('Parsing image files... ')
    for n in range(len(sub_ids)):

        image_path = image_path_list[sub_ids[n]]
        sub_name = image_path.split('/')[-1].split('.')[0]
        subject_names.append(sub_name)
        logging.info('Doing: %s' % sub_name)

        # ================================    
        # read the original mhd image, in order to extract pixel resolution information
        # ================================    
        image_mhd = sitk.ReadImage(image_path)
        pixel_size = image_mhd.GetSpacing()
        logging.info("Original resolution: " + str(pixel_size))
        px_list.append(float(pixel_size[0]))
        py_list.append(float(pixel_size[1]))
        pz_list.append(float(pixel_size[2]))

        img = sitk.GetArrayFromImage(image_mhd)
        logging.info('image stats before norm (min, max, mean): ' + str(np.min(img)) + ', ' + str(np.max(img)) + ', ' + str(np.mean(img)))
            
        # ================================
        # save as nifti, this sets the affine transformation as an identity matrix
        # ================================
        img = io.imread(image_path, plugin='simpleitk')
        seg = io.imread(input_folder + sub_name + '_segmentation.mhd', plugin='simpleitk')
        utils_data.save_nii(img_path = preprocessing_folder + 'PROMISE_nii/' + sub_name + '_img.nii.gz', data = img, affine = np.eye(4))
        utils_data.save_nii(img_path = preprocessing_folder + 'PROMISE_nii/' + sub_name + '_lbl.nii.gz', data = seg, affine = np.eye(4))
        
        # ================================
        # do bias field correction
        # ================================
        input_img = preprocessing_folder + 'PROMISE_nii/' + sub_name + '_img.nii.gz'
        output_img = preprocessing_folder + 'PROMISE_nii/' + sub_name + '_img_n4.nii.gz'
        # If bias corrected image does not exist, do it now
        if os.path.isfile(output_img):
            img = utils_data.load_nii(img_path = output_img)[0]
        else:
            # subprocess.call(["/cluster/home/nkarani/softwares/N4/N4_th", input_img, output_img])
            logging.info('correcting bias field... ')
            utils_data.correct_bias_field(input_img, output_img)
            img = utils_data.load_nii(img_path = output_img)[0]

        # ==================
        # normalize image intensities
        # ==================
        img = utils_data.normalize_intensities(img)
        logging.info('image stats before norm (min, max, mean): ' + str(np.min(img)) + ', ' + str(np.max(img)) + ', ' + str(np.mean(img)))

        # ================================    
        # read the labels from nii
        # ================================   
        logging.info('reading segmentation file...') 
        lbl = utils_data.load_nii(img_path = preprocessing_folder + 'PROMISE_nii/' + sub_name + '_lbl.nii.gz')[0]
        
        # ================================
        # save original label shape (orig image and labels are (z,x,y))
        # ================================
        nx_list.append(lbl.shape[1])
        ny_list.append(lbl.shape[2])
        nz_list.append(lbl.shape[0])
        logging.info('img.shape: %s', str(img.shape))
        logging.info('lbl.shape: %s', str(lbl.shape))

        # ================================
        # rescale resolution and crop / pad slice wise
        # ================================
        # pixel size gives resolution as (z, x, y)
        scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1]]
        image = np.zeros((nx, ny, img.shape[0]), dtype=np.float32)
        label = np.zeros((nx, ny, img.shape[0]), dtype=np.uint8)
        
        for zz in range(img.shape[0]):

            # For some subjects, rescaling directly to the target resolution (0.625) leads to faultily rescaled labels (all pixels get the value 0)
            # Not sure what is causing this.
            # Using this intermediate scaling as a workaround.
            if int(sub_name[4:6]) in [26, 27, 28, 29, 30, 31, 32]:
                logging.info('rescaling workaround for this subject...')
                scale_vector_tmp1 = [pixel_size[0] / 0.65, pixel_size[1] / 0.65]
                img_rescaled = transform.rescale(np.squeeze(img[zz, :, :]), scale_vector_tmp1, order=1, preserve_range=True, multichannel=False, mode = 'constant')
                lbl_rescaled = transform.rescale(np.squeeze(lbl[zz, :, :]), scale_vector_tmp1, order=0, preserve_range=True, multichannel=False, mode='constant')
                scale_vector_tmp2 = [0.65 / target_resolution[0], 0.65 / target_resolution[1]]
                img_rescaled = transform.rescale(img_rescaled, scale_vector_tmp2, order=1, preserve_range=True, multichannel=False, mode = 'constant')
                lbl_rescaled = transform.rescale(lbl_rescaled, scale_vector_tmp2, order=0, preserve_range=True, multichannel=False, mode='constant')
            else:
                img_rescaled = transform.rescale(np.squeeze(img[zz, :, :]), scale_vector, order=1, preserve_range=True, multichannel=False, mode = 'constant')
                lbl_rescaled = transform.rescale(np.squeeze(lbl[zz, :, :]), scale_vector, order=0, preserve_range=True, multichannel=False, mode='constant')

            image[:, :, zz] = utils_data.crop_or_pad(img_rescaled, nx, ny)
            label[:, :, zz] = utils_data.crop_or_pad(lbl_rescaled, nx, ny)
        
        # ================================
        # only binary segmentation
        # ================================
        label[label != 0] = 1

        # ==================
        # write image and label to hdf5 file
        # ==================
        _write_range_to_hdf5(data,
                             image,
                             label,
                             counter_from,
                             counter_from + image.shape[-1])
        
        counter_from = counter_from + image.shape[-1]

    # Write the small datasets
    hdf5_file.create_dataset('nx', data=np.asarray(nx_list, dtype=np.uint16))
    hdf5_file.create_dataset('ny', data=np.asarray(ny_list, dtype=np.uint16))
    hdf5_file.create_dataset('nz', data=np.asarray(nz_list, dtype=np.uint16))
    hdf5_file.create_dataset('depths', data=np.asarray(nz_list, dtype=np.uint16))
    hdf5_file.create_dataset('px', data=np.asarray(px_list, dtype=np.float32))
    hdf5_file.create_dataset('py', data=np.asarray(py_list, dtype=np.float32))
    hdf5_file.create_dataset('pz', data=np.asarray(pz_list, dtype=np.float32))
    hdf5_file.create_dataset('subject_names', data=np.asarray(subject_names, dtype="S20"))
    
    # After test train loop:
    hdf5_file.close()

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
def load_dataset(data_orig_path,
                 data_proc_path,
                 sub_dataset, # UCL / HK / BIDMC
                 train_test_val,
                 cv_fold,
                 size,
                 target_resolution,
                 force_overwrite=False):

    size_str = '_size_' + '_'.join([str(i) for i in size])
    res_str = '_res_' + '_'.join([str(i) for i in target_resolution])
    data_file_name = sub_dataset + '_cv' + str(cv_fold) + '_' + str(train_test_val) + size_str + res_str + '.hdf5'
    data_file_path = os.path.join(data_proc_path, data_file_name)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(data_orig_path,
                     data_proc_path,
                     data_file_path,
                     sub_dataset,
                     train_test_val,
                     cv_fold,
                     size,
                     target_resolution)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')

# ==========================================================
# loads image and label without any preprocessing
# ==========================================================
def load_without_preproc(data_proc_path,
                         subject_name):

    nifti_img_path = data_proc_path + 'PROMISE_nii/' + subject_name
    image = utils_data.load_nii(img_path = nifti_img_path + '_img.nii.gz')[0]
    label = utils_data.load_nii(img_path = nifti_img_path + '_lbl.nii.gz')[0]
                         
    return image, label

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

# ===============================================================
# ===============================================================
if __name__ == '__main__':

    # setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    sub_dataset = 'BIDMC'
    cv_fold = 1
    train_test_val = 'test'
    size = (256, 256)
    data_orig_path = '/data/vision/polina/users/nkarani/data/segmentation/prostate/original/promise/'
    data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/prostate/'

    data = load_dataset(data_orig_path,
                        data_proc_path,
                        sub_dataset,
                        train_test_val,
                        cv_fold,
                        size,
                        target_resolution = (0.625, 0.625),
                        force_overwrite=False)

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
        plt.colorbar()
        plt.subplot(2, 8, s + 9, xticks=[], yticks=[])
        plt.imshow(np.rot90(normalize_img_for_vis(slice_label),k), cmap = 'gray')
        plt.colorbar()

    savepath = data_proc_path + sub_dataset + '_cv_' + str(cv_fold) + '_' + train_test_val + '.png'
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()