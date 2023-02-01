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

# ==================================================
# 2 settings: small (5) and large (15) training dataset
# Test set is the same for all settings -> 10 images
# Val set is the same for all settings -> 5 images
# ==================================================
def get_train_test_val_split_ids(cv_fold_num):

    train_test_val_split_ids = {}

    if cv_fold_num == 1: # 'small training dataset'
        train_test_val_split_ids['test'] = np.arange(0, 10, 1).tolist()
        train_test_val_split_ids['validation'] = np.arange(10, 12, 1).tolist()
        train_test_val_split_ids['train'] = np.arange(15, 20, 1).tolist()

    elif cv_fold_num == 10: # 'small training dataset' | different data points
        train_test_val_split_ids['test'] = np.arange(0, 10, 1).tolist()
        train_test_val_split_ids['validation'] = np.arange(12, 14, 1).tolist()
        train_test_val_split_ids['train'] = np.arange(20, 25, 1).tolist()

    elif cv_fold_num == 2: # 'large training dataset'
        train_test_val_split_ids['test'] = np.arange(0, 10, 1).tolist()
        train_test_val_split_ids['validation'] = np.arange(10, 15, 1).tolist()
        train_test_val_split_ids['train'] = np.arange(15, 30, 1).tolist()

    elif cv_fold_num == 3: # 'smaller training dataset'
        train_test_val_split_ids['test'] = np.arange(0, 10, 1).tolist()
        train_test_val_split_ids['validation'] = np.arange(10, 12, 1).tolist()
        train_test_val_split_ids['train'] = np.arange(15, 17, 1).tolist()

    elif cv_fold_num == 30: # 'smaller training dataset' | different data points
        train_test_val_split_ids['test'] = np.arange(0, 10, 1).tolist()
        train_test_val_split_ids['validation'] = np.arange(12, 14, 1).tolist()
        train_test_val_split_ids['train'] = np.arange(17, 19, 1).tolist()

    return train_test_val_split_ids

# ===============================================================
# ===============================================================
def count_total_slices(folder_list, sub_ids):

    num_slices = 0

    for n in range(len(sub_ids)):
        subject_folder = folder_list[sub_ids[n]]
        logging.info(subject_folder[-15:])
        for _, _, fileList in os.walk(subject_folder):
            for filename in fileList:
                if filename.lower().endswith('.dcm'):  # check whether the file's DICOM
                    num_slices += 1

    return num_slices

# ===============================================================
# Get folder names of all subjects of this dataset
# ===============================================================
def get_patient_folders(image_folder, sub_dataset):

    folder_list = []
    for folder in os.listdir(image_folder):    
        if os.path.isdir(os.path.join(image_folder, folder)):
            series_id = int(folder.split('-')[1])
            patient_id = int(folder.split('-')[2])
            if series_id == 1:
                if sub_dataset == 'RUNMC' and patient_id > 30:
                    continue
                if sub_dataset == 'BMC' and patient_id == 55:
                    continue
                folder_list.append(os.path.join(image_folder, folder))

    return folder_list

# ===============================================================
# ===============================================================
def prepare_data(input_folder,
                 preprocessing_folder,
                 output_file,
                 sub_dataset,
                 train_test_val,
                 cv_fold_num,
                 size,
                 target_resolution):

    # =======================
    # =======================
    if sub_dataset == 'RUNMC':
        image_folder = input_folder + 'Prostate-3T/Images/'
        label_folder = input_folder + 'Prostate-3T/Labels/'
    elif sub_dataset == 'BMC':
        image_folder = input_folder + 'PROSTATE-DIAGNOSIS/Images/'
        label_folder = input_folder + 'PROSTATE-DIAGNOSIS/Labels/'

    # =======================
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # =======================
    # =======================
    logging.info('Counting files and parsing meta data...')
    folder_list = get_patient_folders(image_folder, sub_dataset)

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
        subject_names.append(sub_name)
        logging.info('Doing: %s' % sub_name)

        # ============
        # Make a list of all dicom files in this folder
        # ============
        listFilesDCM = []  # create an empty list
        for dirName, _, fileList in os.walk(subject_folder):
            for filename in fileList:
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    listFilesDCM.append(os.path.join(dirName, filename))

        # ============
        # Get a reference dicom file and extract info such as number of rows, columns, and slices (along the Z axis)
        # ============
        RefDs = dicom.read_file(listFilesDCM[0])
        ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(listFilesDCM))
        logging.info('PixelDims: %s' % str(ConstPixelDims))
        pixel_size = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
        logging.info('PixelSpacing: %s' % str(pixel_size))
        px_list.append(float(RefDs.PixelSpacing[0]))
        py_list.append(float(RefDs.PixelSpacing[1]))
        pz_list.append(float(RefDs.SliceThickness))        

        # ============
        # The image array is sized based on 'ConstPixelDims'
        # ============
        img = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

        # ============
        # loop through all the DICOM files and populate the image array
        # ============
        for filenameDCM in listFilesDCM:
            ds = dicom.read_file(filenameDCM)
            img[:, :, ds.InstanceNumber - 1] = ds.pixel_array
        logging.info('image stats before norm (min, max, mean): ' + str(np.min(img)) + ', ' + str(np.max(img)) + ', ' + str(np.mean(img)))
            
        # ================================
        # save as nifti, this sets the affine transformation as an identity matrix
        # ================================    
        nifti_img_path = preprocessing_folder + 'NCI_nii/' + sub_name
        logging.info('saving image as nifti...')
        utils_data.save_nii(img_path = nifti_img_path + '_img.nii.gz',
                            data = img,
                            affine = np.eye(4))

        # ================================
        # do bias field correction
        # ================================
        input_img = nifti_img_path + '_img.nii.gz'
        output_img = nifti_img_path + '_img_n4.nii.gz'
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
        # read the labels
        # ================================   
        logging.info('reading segmentation file...') 
        lbl_path = os.path.join(label_folder, sub_name + '.nrrd')
        lbl, _ = nrrd.read(lbl_path)

        # fix swap axis
        lbl = np.swapaxes(lbl, 0, 1)

        # ================================ 
        # https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures
        # A competitor reported an issue with case ProstateDx-01-0055, which has a dimension mismatch.
        # The segmentation has dimensions 400x400x23 whereas the DICOM image series have dimensions of 400x400x34.
        # We checked the case and indeed the dimensions seem to not correspond on Z (23 vs 34); however, the labels are properly spatially placed.
        # We don't currently see a problem with using the case. 
        # ================================ 
        if sub_name == 'ProstateDx-01-0055':
            lbl_tmp = np.zeros(shape = img.shape, dtype = lbl.dtype)
            lbl_tmp[:, :, :lbl.shape[2]] = lbl
            lbl = lbl_tmp
        
        # ================================
        # save as nifti, this sets the affine transformation as an identity matrix
        # ================================    
        logging.info('saving label as nifti...') 
        utils_data.save_nii(img_path = nifti_img_path + '_lbl.nii.gz', data = lbl, affine = np.eye(4))
        
        nx_list.append(lbl.shape[0])
        ny_list.append(lbl.shape[1])
        nz_list.append(lbl.shape[2])
        logging.info('img.shape: %s', str(img.shape))
        logging.info('lbl.shape: %s', str(lbl.shape))

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
                 sub_dataset, # RUNMC / BMC
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

    nifti_img_path = data_proc_path + 'NCI_nii/' + subject_name
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

    sub_dataset = 'BMC'
    cv_fold = 10
    train_test_val = 'validation'
    size = (256, 256)

    data_orig_path = '/data/vision/polina/users/nkarani/data/segmentation/prostate/original/nci/'
    data_proc_path = '/data/vision/polina/users/nkarani/projects/crael/seg/data/prostate/'

    data = load_dataset(data_orig_path,
                        data_proc_path,
                        sub_dataset,
                        train_test_val,
                        cv_fold,
                        size,
                        target_resolution = (0.625, 0.625),
                        force_overwrite=True)

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