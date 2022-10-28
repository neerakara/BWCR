# ==================================================
# import stuff
# ==================================================
import numpy as np
import os
import torch
from monai.networks.utils import one_hot
from skimage.filters import gaussian
from skimage import transform
import scipy.ndimage.interpolation
import logging

DEBUGGING = 1

# ==================================================
# ==================================================
def join(a, b):
    return os.path.join(a, b)

# ==================================================
# linear scaling
# ==================================================
def normalize_intensities(array, percentile_min = 2, percentile_max = 98):
    
    low = np.percentile(array, percentile_min)
    high = np.percentile(array, percentile_max)
    normalized_array = (array - low) / (high-low)
    normalized_array[normalized_array < 0.0] = 0.0
    normalized_array[normalized_array > 1.0] = 1.0

    return normalized_array

# ==================================================
# sample a random batch
# ==================================================
def get_batch(images,
              labels,
              batch_size,
              batch_type = 'random',
              start_idx = -1):

    num_slices = images.shape[-1]

    if batch_type == 'sequential' and start_idx != -1:
        batch_indices = np.linspace(start_idx, start_idx + batch_size - 1, batch_size).astype(np.uint8)
    elif batch_type == 'random':
        batch_indices = np.random.randint(0, num_slices, batch_size)

    x_batch = images[:, :, batch_indices]
    y_batch = labels[:, :, batch_indices]

    # swap axes to bring batch dimension from the back to the front
    x_batch = np.swapaxes(np.swapaxes(x_batch, 2, 1), 1, 0)
    y_batch = np.swapaxes(np.swapaxes(y_batch, 2, 1), 1, 0)

    # add channel axis
    x_batch = np.expand_dims(x_batch, axis = 1)
    y_batch = np.expand_dims(y_batch, axis = 1)

    return x_batch, y_batch

def get_number_of_frames_with_fg(y):

    num_fg = 0
    for n in range(y.shape[0]):
        if len(np.unique(y[n,0,:,:])) > 1:
            num_fg = num_fg + 1
    return num_fg

def make_torch_tensors_and_send_to_device(inputs, labels, device, num_labels):
    inputs = torch.from_numpy(inputs)
    labels = torch.from_numpy(labels)
    inputs = inputs.to(device, dtype = torch.float)
    labels = labels.to(device, dtype = torch.float)
    # https://docs.monai.io/en/stable/_modules/monai/networks/utils.html
    labels_one_hot = one_hot(labels = labels, num_classes = num_labels)

    return inputs, labels_one_hot

# ==================================================
# data augmentation
# images and labels created by 'get_batch' will be passed to this function
# [batch_size, num_channels, height, width]
# ==================================================
def transform_images_and_labels(images, labels, data_aug_prob):
    
    # taken from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8995481
    # Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation, TMI 2020
    transform_params = {}
    # probability of each transform
    transform_params['prob'] = data_aug_prob
    # intensity
    transform_params['gamma_min'] = 0.5
    transform_params['gamma_max'] = 4.5
    transform_params['bright_min'] = -0.1
    transform_params['bright_max'] = 0.1
    transform_params['int_scale_min'] = 0.9
    transform_params['int_scale_max'] = 1.1
    # bias field
    # quality
    transform_params['blur_min'] = 0.25
    transform_params['blur_max'] = 1.5
    transform_params['sharpen_min'] = 10.0
    transform_params['sharpen_max'] = 30.0
    transform_params['noise_min'] = 0.1
    transform_params['noise_max'] = 0.2
    # geometric
    transform_params['trans_min'] = -10.0
    transform_params['trans_max'] = 10.0
    transform_params['rot_min'] = -20.0
    transform_params['rot_max'] = 20.0
    transform_params['scale_min'] = 0.4
    transform_params['scale_max'] = 1.6
    # deformation field | figure out how to do this is an invertible way, with few parameters

    n_images = images.shape[0]
    transformed_images = np.copy(images)
    transformed_labels = np.copy(labels)

    if DEBUGGING: logging.info("======================== new batch ========================")
    for n in range(n_images):

        if DEBUGGING: logging.info("============ new 2d image ============")

        # ===================================
        # intensity transformations
        # ===================================
        # 1. gamma contrast
        if np.random.rand() < transform_params['prob']:
            transformed_images[n,0,:,:] = gamma(transformed_images[n,0,:,:], transform_params)

        # 2. intensity scaling and shift (brightness)
        if np.random.rand() < transform_params['prob']:
            transformed_images[n,0,:,:] = scaleshift(transformed_images[n,0,:,:], transform_params)

        # ===================================
        # geometry
        # ===================================
        # 1. translation
        if np.random.rand() < transform_params['prob']:
            transformed_images[n,0,:,:], transformed_labels[n, 0, :, :], tx, ty = translate(transformed_images[n,0,:,:],
                                                                                            transformed_labels[n,0,:,:],
                                                                                            transform_params)
        else:
            tx = 0.0
            ty = 0.0
        # 2. rotation
        if np.random.rand() < transform_params['prob']:
            transformed_images[n,0,:,:], transformed_labels[n, 0, :, :], theta = rotate(transformed_images[n,0,:,:],
                                                                                        transformed_labels[n,0,:,:],
                                                                                        transform_params)
        else:
            theta = 0.0
        # 3. scaling
        if np.random.rand() < transform_params['prob']:
            transformed_images[n,0,:,:], transformed_labels[n, 0, :, :], sc = scale(transformed_images[n,0,:,:],
                                                                                    transformed_labels[n,0,:,:],
                                                                                    transform_params)
        else:
            sc = 1.0

        # ===================================
        # image quality
        # ===================================
        # 1. blur
        if np.random.rand() < transform_params['prob']:  
            transformed_images[n,0,:,:] = blur(transformed_images[n,0,:,:], transform_params)
        # 2. sharpen
        if np.random.rand() < transform_params['prob']:  
            transformed_images[n,0,:,:] = sharpen(transformed_images[n,0,:,:], transform_params)
        # 3. noise
        if np.random.rand() < transform_params['prob']:  
            transformed_images[n,0,:,:] = noise(transformed_images[n,0,:,:], transform_params)          
            
    return transformed_images, transformed_labels, tx, ty, theta, sc

def crop_or_pad(slice, nx, ny):
    x, y = slice.shape
    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2
    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]
    return slice_cropped

def sample_from_uniform(a,b):
    return np.round(np.random.uniform(a, b), 2)

def gamma(image, params):
    c = sample_from_uniform(params['gamma_min'], params['gamma_max'])
    if DEBUGGING == 1: logging.info('doing gamma ' + str(c))
    return image**c

def scaleshift(image, params):
    s = sample_from_uniform(params['int_scale_min'], params['int_scale_max'])
    b = sample_from_uniform(params['bright_min'], params['bright_max'])
    if DEBUGGING == 1: logging.info('doing scaleshift ' + str(s) + ', ' + str(b))
    return image * s + b

def translate(image, label, params):
    tx = sample_from_uniform(params['trans_min'], params['trans_max'])
    ty = sample_from_uniform(params['trans_min'], params['trans_max'])
    if DEBUGGING == 1: logging.info('doing translation ' + str(tx) + ', ' + str(ty))
    translated_image = scipy.ndimage.interpolation.shift(image, shift = (tx, ty), order = 1)
    translated_label = scipy.ndimage.interpolation.shift(label, shift = (tx, ty), order = 0)
    return translated_image, translated_label, tx, ty

def rotate(image, label, params):
    theta = sample_from_uniform(params['rot_min'], params['rot_max'])    
    if DEBUGGING == 1: logging.info('doing rotation ' + str(theta))
    n_x, n_y = image.shape[0], image.shape[1]
    rotated_image = crop_or_pad(scipy.ndimage.interpolation.rotate(image, reshape = False, angle = theta, axes = (1, 0), order = 1), n_x, n_y)
    rotated_label = crop_or_pad(scipy.ndimage.interpolation.rotate(label, reshape = False, angle = theta, axes = (1, 0), order = 0), n_x, n_y)
    return rotated_image, rotated_label, theta

def scale(image, label, params):
    s = sample_from_uniform(params['scale_min'], params['scale_max'])
    if DEBUGGING == 1: logging.info('doing scaling ' + str(s))
    n_x, n_y = image.shape[0], image.shape[1]                
    scaled_image = crop_or_pad(transform.rescale(image, s, order = 1, preserve_range = True, mode = 'constant'), n_x, n_y)
    scaled_label = crop_or_pad(transform.rescale(label, s, order = 0, preserve_range = True, anti_aliasing = False, mode = 'constant'), n_x, n_y)
    return scaled_image, scaled_label, s

def blur(image, params):   
    k = sample_from_uniform(params['blur_min'], params['blur_min'])
    if DEBUGGING == 1: logging.info('doing blurring ' + str(k))
    return gaussian(image, sigma = k)

def sharpen(image, params):
    image1 = blur(image, params)
    image2 = blur(image, params)
    a = sample_from_uniform(params['sharpen_min'], params['sharpen_max'])
    if DEBUGGING == 1: logging.info('doing sharpening ' + str(a))
    return image1 + (image1 - image2) * a

def noise(image, params):
    if DEBUGGING == 1: logging.info('adding noise')
    n = np.random.normal(params['noise_min'], params['noise_max'], size = image.shape)
    return image + n