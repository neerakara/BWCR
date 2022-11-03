# ==================================================
# import stuff
# ==================================================
import nibabel as nib
import numpy as np
import os
import utils_data
import logging

DEBUGGING = 0

# ==================================================
# get image and label paths
# for now, only considering subjects containing only one time point labelled in the time series
# ==================================================
def get_image_and_label_paths(data_path, sub_dataset):

    sub_dataset_path = data_path + sub_dataset + '/'
    files_subdataset = os.listdir(sub_dataset_path)

    sub_ids = []
    for file in files_subdataset:
        if 'segmentation' not in file:
            sub_ids.append(file[4:6])
    sub_ids = np.sort(np.array(sub_ids))

    image_paths = []
    label_paths = []
    for n in range(sub_ids.shape[0]):
        image_paths.append(sub_dataset_path + 'Case' + str(sub_ids[n]) + '.nii.gz')
        label_paths.append(sub_dataset_path + 'Case' + str(sub_ids[n]) + '_segmentation.nii.gz')

    return image_paths, label_paths

# ==================================================
# subdatasets: BIDMC / BMC / HK / I2CVB / RUNMC / UCL
# ==================================================
def get_train_test_val_split_ids(sub_dataset, cv_fold):

    train_test_val_split_ids = {}

    if sub_dataset == 'RUNMC':

        if cv_fold == 1: # 'small training dataset'
            train_test_val_split_ids['test'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            train_test_val_split_ids['train'] = [10, 11, 12, 13, 14]
            train_test_val_split_ids['validation'] = [25, 26, 27, 28, 29]

        elif cv_fold == 2: # 'large training dataset'
            train_test_val_split_ids['test'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            train_test_val_split_ids['train'] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
            train_test_val_split_ids['validation'] = [25, 26, 27, 28, 29]

    elif sub_dataset == 'BMC': 
        train_test_val_split_ids['test'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    elif sub_dataset == 'HK': 
        train_test_val_split_ids['test'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    elif sub_dataset == 'I2CVB': 
        train_test_val_split_ids['test'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    elif sub_dataset == 'UCL': 
        train_test_val_split_ids['test'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    return train_test_val_split_ids

# ==================================================
# Loads the image / label from the source file (e.g. nii) as a numpy array
# ==================================================
def load_images_and_labels(data_path,
                           sub_dataset,
                           train_test_val = 'train',
                           cv_fold = 1):

    image_paths_all, label_paths_all = get_image_and_label_paths(data_path, sub_dataset)
    
    train_test_val_split_ids = get_train_test_val_split_ids(sub_dataset, cv_fold)

    sub_ids = train_test_val_split_ids[train_test_val]
    
    image_paths = []
    label_paths = []

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

        # add to 'images' and 'labels'
        if n == 0:
            images = image
            labels = label
        else:
            images = np.concatenate((images, image), axis=-1)
            labels = np.concatenate((labels, label), axis=-1)

        image_paths.append(image_path)
        label_paths.append(label_path)

        if DEBUGGING == 1:
            print(images.shape)
            print(labels.shape)
    
    return images, labels, image_paths, label_paths