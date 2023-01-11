# ==================================================
# import stuff
# ==================================================
import numpy as np
import os
import torch
from monai.networks.utils import one_hot
from skimage.filters import gaussian
from skimage import transform
from skimage.transform import rescale
import scipy.ndimage.interpolation
import logging
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.transforms as tt
import nibabel as nib
import SimpleITK as sitk

DEBUGGING = 0

# ==================================================
# Shortcut to load a nifti file
# ==================================================
def load_nii(img_path):

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

# ===================================================
# Shortcut to save a nifti file
# ===================================================
def save_nii(img_path, data, affine, header=None):
    if header == None:
        nimg = nib.Nifti1Image(data, affine=affine)
    else:
        nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

# ===================================================
# Remove bias field
# ===================================================
# https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
def correct_bias_field(inputpath, outputpath):
    
    inputImage = sitk.ReadImage(inputpath, sitk.sitkFloat32)
    image = inputImage
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    shrinkFactor = 1
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4
    corrected_image = corrector.Execute(image, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)
    sitk.WriteImage(corrected_image_full_resolution, outputpath)
    return 1

# ========================
# ========================
def rescale_and_crop(volume,
                     scale,
                     size,
                     order):
    
    vol = []
    for zz in range(volume.shape[0]):               
        img = np.squeeze(volume[zz, :, :])
        img_rescaled = rescale(img, scale, order = order, preserve_range = True, multichannel = False, mode = 'constant', anti_aliasing = False)
        img_rescaled_cropped = crop_or_pad(img_rescaled, size[0], size[1])
        vol.append(img_rescaled_cropped)
    
    return np.array(vol)

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
# linear scaling for FLAIR images
# ==================================================
def normalize_intensities_flair(array, percentile_min = 2):

    low = np.percentile(array, percentile_min)
    histogram, bin_edges = np.histogram(array, bins=512)
    high = bin_edges[np.argsort(histogram)[-2]]
    normalized_array = 0.5 * (array - low) / (high-low)

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

# ==================================================
# ==================================================
def get_number_of_frames_with_fg(y):

    num_fg = 0
    for n in range(y.shape[0]):
        if len(np.unique(y[n,0,:,:])) > 1:
            num_fg = num_fg + 1
    return num_fg

# ==================================================
# ==================================================
def make_label_onehot(labels, num_labels):
    # https://docs.monai.io/en/stable/_modules/monai/networks/utils.html
    labels_one_hot = one_hot(labels = labels, num_classes = num_labels)
    return labels_one_hot

# ==================================================
# ==================================================
def torch_and_send_to_device(array, device):
    array = torch.from_numpy(array)
    array = array.to(device, dtype = torch.float)
    return array

# ==================================================
# ==================================================
def get_transform_params(data_aug_prob, device):
    # taken from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8995481
    # Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation, TMI 2020
    transform_params = {}
    # probability of each transform
    transform_params['prob'] = data_aug_prob
    transform_params['device'] = device # required for generating random noise and sending it to gpu
    # intensity
    transform_params['gamma_min'] = 0.5
    transform_params['gamma_max'] = 2.0
    transform_params['bright_min'] = -0.1
    transform_params['bright_max'] = 0.1
    transform_params['int_scale_min'] = 0.9
    transform_params['int_scale_max'] = 1.1
    # bias field
    # quality
    transform_params['blur_min'] = 0 # 0.25
    transform_params['blur_max'] = 5 # 1.5
    transform_params['sharpen_min'] = 0.1
    transform_params['sharpen_max'] = 0.3
    transform_params['noise_min'] = 0.01
    transform_params['noise_max'] = 0.1
    # geometric
    transform_params['trans_min'] = -10.0
    transform_params['trans_max'] = 10.0
    transform_params['rot_min'] = -20.0
    transform_params['rot_max'] = 20.0
    transform_params['scale_min'] = 0.75
    transform_params['scale_max'] = 1.25
    transform_params['shear_min'] = -5.0
    transform_params['shear_max'] = 5.0
    transform_params['sigma_min'] = 10.0
    transform_params['sigma_max'] = 13.0
    transform_params['alpha_min'] = 0.0
    transform_params['alpha_max'] = 1000.0

    return transform_params

# ==================================================
# ==================================================
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

# ==================================================
# ==================================================
def get_param_range(pmin, pmax, num_values):
    delta = (pmax-pmin) / (num_values-1)
    return np.arange(pmin, pmax + delta, delta).tolist()

# ==================================================
# ==================================================
def apply_intensity_transform_fixed(img,
                                    device,
                                    transform_number,
                                    num_transforms):
        
    params = get_transform_params(0.5, device)
    gamma_params = get_param_range(params['gamma_min'], params['gamma_max'], num_transforms)
    scale_params = get_param_range(params['int_scale_min'], params['int_scale_max'], num_transforms)
    shift_params = get_param_range(params['bright_min'], params['bright_max'], num_transforms)
    blur_params_tmp = get_param_range(params['blur_min'], params['blur_max'], num_transforms)
    blur_params = [round(item) for item in blur_params_tmp]
    sharp_params = get_param_range(params['sharpen_min'], params['sharpen_max'], num_transforms)
    noise_params = get_param_range(params['noise_min'], params['noise_max'], num_transforms)

    for b in range(img.shape[0]):
        # 1. gamma contrast
        img[b,0,:,:] = gamma(img[b,0,:,:], params, c = gamma_params[transform_number])
        # 2. intensity scaling and shift (brightness)
        img[b,0,:,:] = scaleshift(img[b,0,:,:], params, s = scale_params[transform_number], b = shift_params[transform_number])
        # 3. image quality - blur
        img[b,0,:,:] = blur(img[b,0,:,:], params, k = blur_params[transform_number])
        # 4. image quality - sharpen
        img[b,0,:,:] = sharpen(img[b,0,:,:], params, a = sharp_params[transform_number], k1 = blur_params[transform_number] + 1, k2 = blur_params[transform_number])
        # 5. image quality - noise
        img[b,0,:,:] = noise(img[b,0,:,:], params, s = noise_params[transform_number])
    
    return img

# ==================================================
# ==================================================
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

# ===============================================================
# crops or pads a volume in the x, y directions.
# size in the z direction is preserved.
# ===============================================================
def crop_or_pad_volume_in_xy(volume, nx, ny):
    x, y, z = volume.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        volume_cropped = volume[x_s:x_s + nx, y_s:y_s + ny, :]
    else:
        volume_cropped = np.zeros((nx, ny, z))
        
        if x <= nx and y > ny:
            volume_cropped[x_c:x_c + x, :, :] = volume[:, y_s:y_s + ny, :]
        
        elif x > nx and y <= ny:
            volume_cropped[:, y_c:y_c + y, :] = volume[x_s:x_s + nx, :, :]
        
        else:
            volume_cropped[x_c:x_c + x, y_c:y_c + y, :] = volume[:, :, :]

    return volume_cropped

# ============================================================
# ============================================================
def sample_from_uniform(a,b):
    return np.round(np.random.uniform(a, b), 2)

# ============================================================
# ============================================================
def sample_from_uniform_integers(a,b):
    return np.random.randint(a, b)

# ============================================================
# ============================================================
def gamma(image, params, c = -1.0):
    if c == -1.0:
        c = sample_from_uniform(params['gamma_min'], params['gamma_max'])
    if DEBUGGING == 1: logging.info('doing gamma ' + str(c))
    return image**c

# ============================================================
# ============================================================
def scaleshift(image, params, s = -1.0, b = -1.0):
    if s == -1.0:
        s = sample_from_uniform(params['int_scale_min'], params['int_scale_max'])
    if b == -1.0:
        b = sample_from_uniform(params['bright_min'], params['bright_max'])
    if DEBUGGING == 1: logging.info('doing scaleshift ' + str(s) + ', ' + str(b))
    return image * s + b

# ============================================================
# ============================================================
def blur(image, params, k = -1.0):
    if k == -1.0:   
        k = sample_from_uniform_integers(params['blur_min'], params['blur_max'])
    if DEBUGGING == 1: logging.info('doing blurring with kernel size ' + str(k))
    return torch.squeeze(TF.gaussian_blur(torch.unsqueeze(torch.unsqueeze(image, 0), 0), kernel_size = 2*k+1))

# ============================================================
# ============================================================
def sharpen(image, params, a = -1.0, k1 = -1.0, k2 = -1.0):
    image1 = blur(image, params, k1)
    image2 = blur(image1, params, k2)
    if a == -1.0:
        a = sample_from_uniform(params['sharpen_min'], params['sharpen_max'])
    if DEBUGGING == 1: logging.info('doing sharpening ' + str(a))
    image_sharp = image1 + (image1 - image2) * a
    image_sharp = (image_sharp - torch.min(image_sharp)) / (torch.max(image_sharp) - torch.min(image_sharp))
    return image_sharp

# ============================================================
# ============================================================
def noise(image, params, s = -1.0):
    if DEBUGGING == 1: logging.info('adding noise')
    if s == -1.0:
        s = sample_from_uniform(params['noise_min'], params['noise_max'])
    noise = torch.normal(0.0, s, size=image.shape).to(params['device'])
    image_noisy = image + noise
    image_noisy = (image_noisy - torch.min(image_noisy)) / (torch.max(image_noisy) - torch.min(image_noisy))
    return image_noisy

# ============================================================
# invert geometric transformations for each prediction in the batch 
# ============================================================
def invert_geometric_transforms(features, t):

    return TF.affine(features,
                     angle=-t[0],
                     translate=[-t[1], -t[2]],
                     scale=1 / t[3],
                     shear=[-t[4], -t[5]],
                     interpolation = transforms.InterpolationMode.BILINEAR)

# ==================================================
# Applies transformation to images and labels
# If t is provided, it should contain affine transformation parameters, and these parameters will be used for applying the geometric transforms
# Otherwise, random geometric transform will be applied.
# ==================================================
def transform_batch(images,
                    labels,
                    data_aug_prob,
                    device,
                    t = 0):
    
    transform_params = get_transform_params(data_aug_prob, device)

    if t != 0:
        images_t, labels_t, t = apply_geometric_transforms_torch(images, labels, transform_params, t = t)
    else:
        images_t, labels_t, t = apply_geometric_transforms_torch(images, labels, transform_params)
    
    for zz in range(images_t.shape[0]):
        images_t[zz, 0, :, :] = apply_intensity_transform(images_t[zz, 0, :, :], transform_params)

    if torch.isnan(torch.mean(images_t)):
        images_t, labels_t, t = transform_batch(images, labels, data_aug_prob, device, t)
    
    return images_t, labels_t, t

# ==================================================
# ==================================================
def apply_geometric_transforms_torch(images,
                                     labels,
                                     params,
                                     t = 0):

    if t == 0:
        t = sample_affine_params(params)
    
    images_t = TF.affine(images,
                         angle=t[0],
                         translate=[t[1], t[2]],
                         scale=t[3],
                         shear=[t[4], t[5]],
                         interpolation = tt.InterpolationMode.BILINEAR)

    labels_t = TF.affine(labels,
                         angle=t[0],
                         translate=[t[1], t[2]],
                         scale=t[3],
                         shear=[t[4], t[5]],
                         interpolation = tt.InterpolationMode.NEAREST)    
    
    return images_t, labels_t, t

# ==================================================
# ==================================================
def apply_geometric_transforms_mask(mask, t):

    return TF.affine(mask,
                     angle=t[0],
                     translate=[t[1], t[2]],
                     scale=t[3],
                     shear=[t[4], t[5]],
                     interpolation = tt.InterpolationMode.NEAREST)

# ==================================================
# ==================================================
def invert_geometric_transforms_mask(mask, t):

    return TF.affine(mask,
                     angle=-t[0],
                     translate=[-t[1], -t[2]],
                     scale=1 / t[3],
                     shear=[-t[4], -t[5]],
                     interpolation = transforms.InterpolationMode.BILINEAR)

# ==================================================
# ==================================================
def rescale_tensor(t, newsize):
    return TF.resize(t, newsize)

# ==================================================
# ==================================================
def sample_affine_params(params):

    # rotation
    if np.random.rand() < params['prob']:
        r = sample_from_uniform(params['rot_min'], params['rot_max'])
    else:
        r = 0.0

    # translation
    if np.random.rand() < params['prob']:
        tx = sample_from_uniform(params['trans_min'], params['trans_max'])
        ty = sample_from_uniform(params['trans_min'], params['trans_max'])
    else:
        tx = 0.0
        ty = 0.0

    # scale
    if np.random.rand() < params['prob']:
        s = sample_from_uniform(params['scale_min'], params['scale_max'])
    else:
        s = 1.0
        
    # shear
    if np.random.rand() < params['prob']:
        sx = sample_from_uniform(params['shear_min'], params['shear_max'])
        sy = sample_from_uniform(params['shear_min'], params['shear_max'])
    else:
        sx = 0.0
        sy = 0.0
    
    return [r, tx, ty, s, sx, sy]