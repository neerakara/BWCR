# ===============================================================
# visualization functions
# ===============================================================
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np
import logging
import skimage.segmentation

# ==========================================================
# ==========================================================
def show_images_labels_predictions(images,
                                   labels,
                                   soft_predictions):
    
    fig = plt.figure(figsize=(18, 24))
    
    tmp_images = np.copy(images.cpu().numpy())
    tmp_labels = np.copy(labels.cpu().numpy())
    tmp_predictions = np.copy(soft_predictions.detach().cpu().numpy())

    # show 4 examples per batch
    for batch_index in range(4):
        ax = fig.add_subplot(4, 3, 3*batch_index + 1, xticks=[], yticks=[])
        plt.imshow(normalize_img_for_vis(tmp_images[batch_index,0,:,:]), cmap = 'gray')
        ax = fig.add_subplot(4, 3, 3*batch_index + 2, xticks=[], yticks=[])
        plt.imshow(normalize_img_for_vis(tmp_labels[batch_index,1,:,:]), cmap = 'gray')
        ax = fig.add_subplot(4, 3, 3*batch_index + 3, xticks=[], yticks=[])
        plt.imshow(normalize_img_for_vis(tmp_predictions[batch_index,1,:,:]), cmap = 'gray')

    return fig

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

# ==========================================================
# ==========================================================
def save_images_and_labels(images1,
                           labels1,
                           images2,
                           labels2,
                           savefilename,
                           nc = 6,
                           nr = 5):

    fig = plt.figure(figsize=(6*nc, 6*nr))
    for c in range(nc):
        
        fig.add_subplot(nr, nc, c + 1, xticks=[], yticks=[])
        plt.imshow(images1[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        
        fig.add_subplot(nr, nc, nc + c + 1, xticks=[], yticks=[])
        plt.imshow(images2[c,0,:,:], cmap = 'gray')
        plt.colorbar()

        fig.add_subplot(nr, nc, 2*nc + c + 1, xticks=[], yticks=[])
        plt.imshow(images1[c,0,:,:] - images2[c,0,:,:], cmap = 'gray')
        plt.colorbar()

        fig.add_subplot(nr, nc, 3*nc + c + 1, xticks=[], yticks=[])
        plt.imshow(labels1[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        
        fig.add_subplot(nr, nc, 4*nc + c + 1, xticks=[], yticks=[])
        plt.imshow(labels2[c,0,:,:], cmap = 'gray')
        plt.colorbar()

    plt.savefig(savefilename, bbox_inches='tight', dpi=50)
    plt.close()

    return 0