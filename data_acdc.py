import os
import numpy as np
import logging
import gc
import h5py
from skimage import transform
import utils_data
import pydicom as dicom
import nrrd
import subprocess
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.use('agg')
import matplotlib.pyplot as plt 
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# ===============================================================
# Get folder names of all subjects of this dataset
# ===============================================================
def get_patient_folders(image_folder,
                        train_test_val):

    folder_list = []
    
    if train_test_val == 'test':
        image_folder = image_folder + 'testing/'

    else:
        image_folder = image_folder + 'training/'
    
    for folder in os.listdir(image_folder):    
        
        if os.path.isdir(os.path.join(image_folder, folder)):
        
            folder_list.append(os.path.join(image_folder, folder))

    return folder_list

# ==================================================
# ==================================================
def get_train_test_val_split_ids(cv_fold_num):

    train_test_val_split_ids = {}
    train_test_val_split_ids['test'] = np.arange(0, 50, 1).tolist()

    # num_train_label = 5 | num_val_label = 5 |  num_val_unlabel = 90
    # ran "np.random.randint(0, 100, 10).tolist()" thrice and fixed values to avoid confusion when running again
    if cv_fold_num == 1:
        idx_train_val = [39, 26, 88, 72, 4, 8, 73, 19, 53, 40]
    elif cv_fold_num == 2:
        idx_train_val = [65, 22, 53, 0, 77, 92, 97, 38, 40, 51]
    elif cv_fold_num == 3:
        idx_train_val = [84, 3, 59, 21, 17, 57, 20, 77, 60, 91]

    train_test_val_split_ids['train'] = idx_train_val[:5]
    train_test_val_split_ids['validation'] = idx_train_val[5:]

    idx_total = np.arange(0, 100, 1).tolist()
    train_test_val_split_ids['train_unsupervised'] = [x for x in idx_total if x not in idx_train_val]

    return train_test_val_split_ids

# ===============================================================
# ===============================================================
def count_total_slices(folder_list, sub_ids):

    num_slices = 0

    for n in range(len(sub_ids)):
    
        subject_folder = folder_list[sub_ids[n]]
    
        for _, _, fileList in os.walk(subject_folder):
    
            for filename in fileList:
        
                if len(filename) == 25:
                    img_path = subject_folder + '/' + filename
                    num_slices += utils_data.load_nii(img_path)[0].shape[-1]

    return num_slices

# ===============================================================
# ===============================================================
def prepare_data(input_folder,
                 preprocessing_folder,
                 output_file,
                 train_test_val,
                 cv_fold_num,
                 size,
                 target_resolution,
                 bias_correct):

    # =======================
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # =======================
    # =======================
    logging.info('Counting files and parsing meta data...')
    folder_list = get_patient_folders(input_folder, train_test_val)

    # get ids for this train / test / validation split
    logging.info('Getting ids of subjects to be read...')
    train_test_val_split_ids = get_train_test_val_split_ids(cv_fold_num)
    sub_ids = train_test_val_split_ids[train_test_val]
    
    # ===============================
    # count number of slices to pre-define dataset size
    # ===============================
    logging.info('Counting dataset size...')
    num_slices = count_total_slices(folder_list, sub_ids)
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

        subject_folder = folder_list[sub_ids[n]]
        sub_name = subject_folder.split('/')[-1]
        logging.info('Doing: %s' % sub_name)
        
        # ================================
        # there must be two labelled volumes for each subject - at ED (frame01) and ES (frameXX) in the cardiac cycle
        # ED is when the heart is full of blood
        # ES is when the heart has pumped out the blood for this cycle
        # ================================
        for _, _, fileList in os.walk(subject_folder):    
            for filename in fileList:
                if len(filename) == 25:
                    if filename == sub_name + '_frame01.nii.gz':
                        cardiac_stage = 'ED'
                    else:
                        cardiac_stage = 'ES'
        
                    subject_names.append(sub_name + cardiac_stage)
                    logging.info(cardiac_stage)

                    # ================================
                    # get orig resolution
                    # ================================
                    img_path = subject_folder + '/' + filename
                    pixel_size = utils_data.load_nii(img_path = img_path)[-1].get_zooms()
                    px_list.append(float(pixel_size[0]))
                    py_list.append(float(pixel_size[1]))
                    pz_list.append(float(pixel_size[2]))

                    # ================================
                    # do bias field correction
                    # ================================
                    img_path_n4 = preprocessing_folder + sub_name + '_' + cardiac_stage + '_n4.nii.gz'
                    if bias_correct == True:
                        # If bias corrected image does not exist, do it now
                        if os.path.isfile(img_path_n4):
                            img = utils_data.load_nii(img_path = img_path_n4)[0]
                        else:
                            logging.info('correcting bias field... ')
                            utils_data.correct_bias_field(img_path, img_path_n4, 'n4_exec')
                            img = utils_data.load_nii(img_path = img_path_n4)[0]
                    else:
                        img = utils_data.load_nii(img_path = img_path)[0]

                    # ==================
                    # normalize image intensities
                    # ==================
                    img = utils_data.normalize_intensities(img)
                    logging.info('image stats before norm (min, max, mean): ' + str(np.min(img)) + ', ' + str(np.max(img)) + ', ' + str(np.mean(img)))

                    # ================================    
                    # read the labels
                    # ================================   
                    logging.info('reading segmentation file...') 
                    lbl_path = subject_folder + '/' + filename.split('.')[0] + '_gt.nii.gz'
                    lbl = utils_data.load_nii(img_path = lbl_path)[0]

                    nx_list.append(lbl.shape[0])
                    ny_list.append(lbl.shape[1])
                    nz_list.append(lbl.shape[2])

                    # ================================
                    # rescale resolution and crop / pad slice wise
                    # ================================
                    scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1]]
                    image = np.zeros((nx, ny, img.shape[2]), dtype=np.float32)
                    label = np.zeros((nx, ny, img.shape[2]), dtype=np.uint8)
                    for zz in range(img.shape[2]):
                        img_rescaled = transform.rescale(np.squeeze(img[:, :, zz]), scale_vector, order=1, preserve_range=True, multichannel=False, mode = 'constant')
                        lbl_rescaled = transform.rescale(np.squeeze(lbl[:, :, zz]), scale_vector, order=0, preserve_range=True, multichannel=False, mode='constant')
                        img_cropped = utils_data.crop_or_pad(img_rescaled, nx, ny)
                        lbl_cropped = utils_data.crop_or_pad(lbl_rescaled, nx, ny)
                        image[:, :, zz] = img_cropped
                        label[:, :, zz] = lbl_cropped
        
                    logging.info('img.shape: %s', str(image.shape))
                    logging.info('lbl.shape: %s', str(label.shape))

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
                 train_test_val,
                 cv_fold,
                 size,
                 target_resolution,
                 bias_correct=True,
                 force_overwrite=False):

    size_str = '_size_' + '_'.join([str(i) for i in size])
    res_str = '_res_' + '_'.join([str(i) for i in target_resolution])
    if bias_correct == False:
        data_file_name = 'cv' + str(cv_fold) + '_' + str(train_test_val) + size_str + res_str + '.hdf5'
    else:
        data_file_name = 'cv' + str(cv_fold) + '_' + str(train_test_val) + size_str + res_str + '_bias_corrected.hdf5'
    data_file_path = os.path.join(data_proc_path, data_file_name)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(data_orig_path,
                     data_proc_path,
                     data_file_path,
                     train_test_val,
                     cv_fold,
                     size,
                     target_resolution,
                     bias_correct)
    else:
        logging.info('Already preprocessed this configuration. Loading ' + data_file_path + ' now!')

    return h5py.File(data_file_path, 'r')

# ==========================================================
# loads image and label without any preprocessing
# ==========================================================
def load_without_preproc(data_path,
                         subject_name):

    sub_name = subject_name[:-2]
    subject_folder = data_path + sub_name + '/'

    for _, _, fileList in os.walk(subject_folder):    
        
        for filename in fileList:
        
            if len(filename) == 25:
        
                if filename == sub_name + '_frame01.nii.gz':
                    cardiac_stage = 'ED'
                else:
                    cardiac_stage = 'ES'
                    
                if cardiac_stage == subject_name[-2:]:
                    image = utils_data.load_nii(img_path = subject_folder + filename)[0]
                    label = utils_data.load_nii(img_path = subject_folder + '/' + filename.split('.')[0] + '_gt.nii.gz')[0]
                    
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

    cv_fold = 3
    train_test_val = 'validation'
    size = (192, 192)
    target_res = (1.33, 1.33)
    bias_correct = True

    data_orig_path = '/data/vision/polina/users/nkarani/data/segmentation/acdc/'
    data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/acdc/'

    data = load_dataset(data_orig_path,
                        data_proc_path,
                        train_test_val,
                        cv_fold,
                        size,
                        target_res,
                        bias_correct,
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

    if bias_correct == False:
        savepath = data_proc_path + '_cv_' + str(cv_fold) + '_' + train_test_val + '.png'
    else:
        savepath = data_proc_path + '_cv_' + str(cv_fold) + '_' + train_test_val + '_bias_correct.png'
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()