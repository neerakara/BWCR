# ==================================================
# import stuff
# ==================================================
import nibabel as nib
import numpy as np
import os
import utils_data

DEBUGGING = 1

# ==================================================
# get image and label paths
# for now, only considering subjects containing only one time point labelled in the time series
# ==================================================
def get_image_and_label_paths(data_path):

    sub_dirs = [d for d in os.listdir(data_path) if os.path.isdir(utils_data.join(data_path, d))]

    image_paths = []
    label_paths = []

    for n in range(len(sub_dirs)):
        
        sub_dir = sub_dirs[n]
        sub_path = utils_data.join(data_path, sub_dir)
        image_names = os.listdir(utils_data.join(sub_path, 'volume'))
        label_names = os.listdir(utils_data.join(sub_path, 'segmentation'))

        # ignore subjects that contain more than one time points labelled.
        if len(image_names) > 1:
            continue

        image_paths.append(utils_data.join(utils_data.join(sub_path, 'volume'), image_names[0]))
        label_paths.append(utils_data.join(utils_data.join(sub_path, 'segmentation'), label_names[0]))

    return image_paths, label_paths

# ==================================================
# ==================================================
def get_train_test_val_split_ids(cv_fold):

    train_test_val_split_ids = {}

    if cv_fold == 1:
        train_test_val_split_ids['train'] = [1, 2, 3, 4, 5]
        train_test_val_split_ids['validation'] = [6, 7, 8, 9, 10]
        train_test_val_split_ids['test'] = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

    elif cv_fold == 2:
        train_test_val_split_ids['train'] = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        train_test_val_split_ids['validation'] = [6, 7, 8, 9, 10]
        train_test_val_split_ids['test'] = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

    return train_test_val_split_ids

# ==================================================
# Loads the image / label from the source file (e.g. nii) as a numpy array
# ==================================================
def load_images_and_labels(data_path,
                           train_test_val = 'train',
                           cv_fold = 1):

    image_paths_all, label_paths_all = get_image_and_label_paths(data_path)
    
    train_test_val_split_ids = get_train_test_val_split_ids(cv_fold = cv_fold)

    sub_ids = train_test_val_split_ids[train_test_val]
    
    image_paths = []
    label_paths = []
    depths = []

    for n in range(len(sub_ids)):
    
        # read image
        image_path = image_paths_all[sub_ids[n]]
        image = nib.load(image_path)
        image = np.array(image.dataobj)
        image = image.astype(float)
    
        # normalize image intensities
        image = utils_data.normalize_intensities(image)

        # read label
        label_path = label_paths_all[sub_ids[n]]
        label = nib.load(label_path)
        label = np.array(label.dataobj)

        if DEBUGGING == 1:
            print(image_path)
            print(image.shape)
            print(label_path)
            print(label.shape)

        # squeeze
        image = np.squeeze(image)
        label = np.squeeze(label)
        depths.append(image.shape[-1])

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

    data = {}
    data['images'] = images
    data['labels'] = labels
    data['image_paths'] = image_paths
    data['label_paths'] = label_paths
    data['depths'] = depths
    
    return images, labels, image_paths, label_paths