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
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

DEBUGGING = 0

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

    if batch_type == 'sequential' and start_idx != -1:
        batch_indices = np.linspace(start_idx, start_idx + batch_size - 1, batch_size).astype(np.uint8)
    elif batch_type == 'random':
        n_slices = images.shape[-1]
        random_indices = np.random.permutation(n_slices)
        batch_indices = np.sort(random_indices[:batch_size])

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

def make_label_onehot(labels, num_labels):
    # https://docs.monai.io/en/stable/_modules/monai/networks/utils.html
    labels_one_hot = one_hot(labels = labels, num_classes = num_labels)
    return labels_one_hot

def make_torch_tensor_and_send_to_device(array, device):
    array = torch.from_numpy(array)
    array = array.to(device, dtype = torch.float)
    return array

# ==================================================
# data augmentation
# images and labels created by 'get_batch' will be passed to this function
# [batch_size, num_channels, height, width]
# ==================================================
def transform_for_data_aug(images,
                           labels,
                           data_aug_prob):
    
    transform_params = get_transform_params(data_aug_prob)
    transformed_images = np.copy(images)
    transformed_labels = np.copy(labels)

    for n in range(images.shape[0]):
        # geometric transformations
        transformed_images[n,0,:,:], transformed_labels[n,0,:,:] = apply_geometric_transforms(transformed_images[n,0,:,:],
                                                                                              transformed_labels[n,0,:,:],
                                                                                              transform_params)
        # intensity transformations
        transformed_images[n,0,:,:] = apply_intensity_transform(transformed_images[n,0,:,:], transform_params)
            
    return transformed_images, transformed_labels

# ==================================================
# data augmentation
# images and labels created by 'get_batch' will be passed to this function
# [batch_size, num_channels, height, width]
# ==================================================
def transform_for_data_cons(images,
                            labels,
                            data_aug_prob):
    
    transform_params = get_transform_params(data_aug_prob)
    transformed_images = np.copy(images)
    transformed_labels = np.copy(labels)
    transformed1_images = np.copy(images)
    transformed2_images = np.copy(images)
    

    for n in range(images.shape[0]):
        # geometric transformations
        transformed_images[n,0,:,:], transformed_labels[n,0,:,:] = apply_geometric_transforms(transformed_images[n,0,:,:],
                                                                                              transformed_labels[n,0,:,:],
                                                                                              transform_params)
        # intensity transformations x 2
        transformed1_images[n,0,:,:] = apply_intensity_transform(transformed_images[n,0,:,:], transform_params)
        transformed2_images[n,0,:,:] = apply_intensity_transform(transformed_images[n,0,:,:], transform_params)
            
    return transformed1_images, transformed2_images, transformed_labels

def get_transform_params(data_aug_prob):
    # taken from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8995481
    # Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation, TMI 2020
    transform_params = {}
    # probability of each transform
    transform_params['prob'] = data_aug_prob
    # intensity
    transform_params['gamma_min'] = 0.5
    transform_params['gamma_max'] = 2.5
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
    transform_params['noise_min'] = 0.01
    transform_params['noise_max'] = 0.1
    # geometric
    transform_params['trans_min'] = -10.0
    transform_params['trans_max'] = 10.0
    transform_params['rot_min'] = -20.0
    transform_params['rot_max'] = 20.0
    transform_params['scale_min'] = 0.4
    transform_params['scale_max'] = 1.6
    transform_params['sigma_min'] = 10.0
    transform_params['sigma_max'] = 13.0
    transform_params['alpha_min'] = 0.0
    transform_params['alpha_max'] = 1000.0

    return transform_params

def apply_geometric_transforms(img, lbl, params):
    # 1. translation
    if np.random.rand() < params['prob']:
        img, lbl = translate(img, lbl, params)
    # 2. rotation
    if np.random.rand() < params['prob']:
        img, lbl = rotate(img, lbl, params)
    # 3. scaling
    if np.random.rand() < params['prob']:
        img, lbl = scale(img, lbl, params)
    # 4. elastic deform
    if np.random.rand() < params['prob']:
        img, lbl = elastic_deform(img, lbl, params)
    return img, lbl

def apply_intensity_transform(img, params):
    # 1. gamma contrast
    if np.random.rand() < params['prob']:
        img = gamma(img, params)
    # 2. intensity scaling and shift (brightness)
    if np.random.rand() < params['prob']:
        img = scaleshift(img, params)
    # 3. image quality - blur
    if np.random.rand() < params['prob']:  
        img = blur(img, params)
    # 4. image quality - sharpen
    if np.random.rand() < params['prob']:  
        img = sharpen(img, params)
    # 5. image quality - noise
    if np.random.rand() < params['prob']:  
        img = noise(img, params)          
    return img

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
    return translated_image, translated_label

def rotate(image, label, params):
    theta = sample_from_uniform(params['rot_min'], params['rot_max'])    
    if DEBUGGING == 1: logging.info('doing rotation ' + str(theta))
    n_x, n_y = image.shape[0], image.shape[1]
    rotated_image = crop_or_pad(scipy.ndimage.interpolation.rotate(image, reshape = False, angle = theta, axes = (1, 0), order = 1), n_x, n_y)
    rotated_label = crop_or_pad(scipy.ndimage.interpolation.rotate(label, reshape = False, angle = theta, axes = (1, 0), order = 0), n_x, n_y)
    return rotated_image, rotated_label

def scale(image, label, params):
    s = sample_from_uniform(params['scale_min'], params['scale_max'])
    if DEBUGGING == 1: logging.info('doing scaling ' + str(s))
    n_x, n_y = image.shape[0], image.shape[1]                
    scaled_image = crop_or_pad(transform.rescale(image, s, order = 1, preserve_range = True, mode = 'constant'), n_x, n_y)
    scaled_label = crop_or_pad(transform.rescale(label, s, order = 0, preserve_range = True, anti_aliasing = False, mode = 'constant'), n_x, n_y)
    return scaled_image, scaled_label

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

# taken from: https://gist.github.com/erniejunior/601cdf56d2b424757de5 
def elastic_deform(image, # 2d
                   label,
                   params,
                   random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    sigma = sample_from_uniform(params['sigma_min'], params['sigma_max'])
    alpha = sample_from_uniform(params['alpha_min'], params['alpha_max'])

    # random_state.rand(*shape) generate an array of image size with random uniform noise between 0 and 1
    # random_state.rand(*shape)*2 - 1 becomes an array of image size with random uniform noise between -1 and 1
    # applying the gaussian filter with a relatively large std deviation (~20) makes this a relatively smooth deformation field, but with very small deformation values (~1e-3)
    # multiplying it with alpha (500) scales this up to a reasonable deformation (max-min:+-10 pixels)
    # multiplying it with alpha (1000) scales this up to a reasonable deformation (max-min:+-25 pixels)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    distored_label = map_coordinates(label, indices, order=0, mode='reflect').reshape(shape)
    
    return distored_image, distored_label

# ============================================================
# invert geometric transformations for each prediction in the batch 
# ============================================================
def invert_geometric_transforms(features, geom_params):
    batch_size = features.shape[0]
    features_cloned = torch.clone(features)
    for n in range(batch_size):
        features_cloned[n,:,:,:] = TF.affine(features[n,:,:,:],
                                             angle = geom_params['thetas'][n], # scipy and TF do rotation in opposite directions by detault!
                                             translate = [-geom_params['trans_x'][n], -geom_params['trans_y'][n]],
                                             scale = 1 / geom_params['scales'][n],
                                             shear = 0.0,
                                             interpolation = transforms.InterpolationMode.BILINEAR)
    return features_cloned