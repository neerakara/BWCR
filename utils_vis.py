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
def save_images_labels(images,
                       labels,
                       savepath):
    
    plt.figure(figsize=(12, 24))
    k=-1
    # show 4 examples per batch
    for batch_index in range(4):
        plt.subplot(4, 2, 2*batch_index + 1, xticks=[], yticks=[])
        plt.imshow(np.rot90(normalize_img_for_vis(images[batch_index,:,:]),k), cmap = 'gray')
        plt.subplot(4, 2, 2*batch_index + 2, xticks=[], yticks=[])
        plt.imshow(np.rot90(normalize_img_for_vis(labels[batch_index,:,:]),k), cmap = 'gray')
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()
    return 0

# ==========================================================
# ==========================================================
def save_images_labels_predictions_soft_and_hard(images,
                                                 labels,
                                                 soft_predictions,
                                                 savepath):

    plt.figure(figsize=(24, 24))
    
    tmp_images = np.copy(images.cpu().numpy())
    tmp_labels = np.copy(labels.cpu().numpy())
    tmp_predictions_soft = np.copy(soft_predictions.detach().cpu().numpy())
    tmp_predictions_soft_tmp = np.copy(tmp_predictions_soft)
    tmp_predictions_hard = (tmp_predictions_soft_tmp[:, 1, :, :] > 0.5).astype(np.float16)

    # show 4 examples per batch
    for batch_index in range(4):
        k=-1
        im = np.rot90(normalize_img_for_vis(tmp_images[batch_index,0,:,:]), k)
        lb = np.rot90(normalize_img_for_vis(tmp_labels[batch_index,1,:,:]), k)
        prs = np.rot90(normalize_img_for_vis(tmp_predictions_soft[batch_index,1,:,:]), k)
        prh = np.rot90(normalize_img_for_vis(tmp_predictions_hard[batch_index,:,:]), k)

        plt.subplot(4, 4, 4*batch_index + 1, xticks=[], yticks=[]); plt.imshow(im, cmap = 'gray'); plt.colorbar()
        plt.subplot(4, 4, 4*batch_index + 2, xticks=[], yticks=[]); plt.imshow(lb, cmap = 'gray'); plt.colorbar()
        plt.subplot(4, 4, 4*batch_index + 3, xticks=[], yticks=[]); plt.imshow(prs, cmap = 'gray'); plt.colorbar()
        plt.subplot(4, 4, 4*batch_index + 4, xticks=[], yticks=[]); plt.imshow(prh, cmap = 'gray'); plt.colorbar()

    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    
    return 0

# ==========================================================
# ==========================================================
def save_results(images,
                 labels,
                 preds,
                 savepath):
    
    # find slice with largest foreground
    zz = np.argmax(np.sum(labels, axis=(0,1)))
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1); plt.imshow(normalize_img_for_vis(images[:,:,zz]), cmap = 'gray')
    plt.subplot(1, 3, 2); plt.imshow(normalize_img_for_vis(labels[:,:,zz]), cmap = 'gray')
    plt.subplot(1, 3, 3); plt.imshow(normalize_img_for_vis(preds[:,:,zz]), cmap = 'gray')
    plt.savefig(savepath, bbox_inches='tight', dpi=50)

    return 0

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
def save_images_and_labels(images,
                           labels,
                           savefilename,
                           normalize = False,
                           rot90_k = -1):
    
    plt.figure(figsize=(12, 24))
    tmp_images = images.detach().cpu().numpy()
    tmp_labels = labels.detach().cpu().numpy()
    # show 4 examples per batch
    for batch_index in range(4):
        if normalize == True:
            im = normalize_img_for_vis(tmp_images[batch_index,0,:,:])
            lb = normalize_img_for_vis(tmp_labels[batch_index,0,:,:])
        else:
            im = tmp_images[batch_index,0,:,:]
            lb = tmp_labels[batch_index,0,:,:]
        if rot90_k != 0:
            im = np.rot90(im, k=rot90_k)
            lb = np.rot90(lb, k=rot90_k)
        plt.subplot(4, 2, 2*batch_index + 1, xticks=[], yticks=[]); plt.imshow(im, cmap = 'gray'); plt.colorbar()
        plt.subplot(4, 2, 2*batch_index + 2, xticks=[], yticks=[]); plt.imshow(lb, cmap = 'gray'); plt.colorbar()
    plt.savefig(savefilename, bbox_inches='tight', dpi=50)
    plt.close()
    return 0

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
def save_images_and_labels_orig_and_transformed(images,
                                                labels,
                                                images1,
                                                labels1,
                                                savefilename,
                                                nc = 6,
                                                nr = 5):

    fig = plt.figure(figsize=(6*nc, 6*nr))
    for c in range(nc):
        
        fig.add_subplot(nr, nc, c + 1, xticks=[], yticks=[])
        plt.imshow(images[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.title('orig image')
        
        fig.add_subplot(nr, nc, nc + c + 1, xticks=[], yticks=[])
        plt.imshow(images1[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.title('transformed image')

        fig.add_subplot(nr, nc, 2*nc + c + 1, xticks=[], yticks=[])
        plt.imshow(images[c,0,:,:] - images1[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.title('orig image - transformed image')

        fig.add_subplot(nr, nc, 3*nc + c + 1, xticks=[], yticks=[])
        plt.imshow(labels[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.title('orig label')
        
        fig.add_subplot(nr, nc, 4*nc + c + 1, xticks=[], yticks=[])
        plt.imshow(labels1[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.title('transformed label')

    plt.savefig(savefilename, bbox_inches='tight', dpi=50)
    plt.close()

    return 0

def save_images_and_labels_orig_and_transformed_two_ways(images,
                                                         labels,
                                                         images1,
                                                         labels1,
                                                         images2,
                                                         labels2,
                                                         savefilename,
                                                         nc = 6,
                                                         nr = 7):

    fig = plt.figure(figsize=(6*nc, 6*nr))
    
    for c in range(nc):
        
        fig.add_subplot(nr, nc, c + 1, xticks=[], yticks=[])
        plt.imshow(images[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.title('orig image')
        
        fig.add_subplot(nr, nc, nc + c + 1, xticks=[], yticks=[])
        plt.imshow(images1[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.title('transform 1 image')

        fig.add_subplot(nr, nc, 2*nc + c + 1, xticks=[], yticks=[])
        plt.imshow(images2[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.title('transform 2 image')

        fig.add_subplot(nr, nc, 3*nc + c + 1, xticks=[], yticks=[])
        plt.imshow(images1[c,0,:,:] - images2[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.title('transform 1 image - transform 2 image')

        fig.add_subplot(nr, nc, 4*nc + c + 1, xticks=[], yticks=[])
        plt.imshow(labels[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.title('orig label')
        
        fig.add_subplot(nr, nc, 5*nc + c + 1, xticks=[], yticks=[])
        plt.imshow(labels1[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.title('transform 1 label')

        fig.add_subplot(nr, nc, 6*nc + c + 1, xticks=[], yticks=[])
        plt.imshow(labels2[c,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.title('transform 2 label')

    plt.savefig(savefilename, bbox_inches='tight', dpi=50)
    plt.close()

    return 0

def save_debug(images,
               labels,
               images1,
               labels1,
               images2,
               labels2,
               pred1,
               pred2,
               savefilename,
               nc = 6,
               nr = 9):

    fig = plt.figure(figsize=(6*nc, 6*nr))
    for c in range(nc):        
        makesubplot(fig, nr, nc, c+1, images[c,0,:,:], 'orig image')
        makesubplot(fig, nr, nc, nc+c+1, images1[c,0,:,:], 'transform 1 image')
        makesubplot(fig, nr, nc, 2*nc+c+1, images2[c,0,:,:], 'transform 2 image')
        makesubplot(fig, nr, nc, 3*nc+c+1, images1[c,0,:,:] - images2[c,0,:,:], 'transform 1 image - transform 2 image')
        makesubplot(fig, nr, nc, 4*nc+c+1, labels[c,0,:,:], 'orig label')
        makesubplot(fig, nr, nc, 5*nc+c+1, labels1[c,0,:,:], 'transform 1 label')
        makesubplot(fig, nr, nc, 6*nc+c+1, labels2[c,0,:,:], 'transform 2 label')
        makesubplot(fig, nr, nc, 7*nc+c+1, pred1[c,0,:,:], 'pred 1')
        makesubplot(fig, nr, nc, 8*nc+c+1, pred2[c,0,:,:], 'pred 2')
    plt.savefig(savefilename, bbox_inches='tight', dpi=50)
    plt.close()
    return 0

def save_heads(images,
               images1,
               images2,
               headsl1_1,
               headsl1_2,
               headsl2_1,
               headsl2_2,
               headsl3_1,
               headsl3_2,
               preds1,
               preds2,
               savefilename,
               nc = 6,
               nr = 11):

    fig = plt.figure(figsize=(6*nc, 6*nr))
    for c in range(nc):        
        makesubplot(fig, nr, nc, c+1, images[c,0,:,:], 'orig image')
        makesubplot(fig, nr, nc, nc+c+1, images1[c,0,:,:], 'transform 1 image')
        makesubplot(fig, nr, nc, 2*nc+c+1, images2[c,0,:,:], 'transform 2 image')
        makesubplot(fig, nr, nc, 3*nc+c+1, headsl1_1[c,0,:,:], 'transform 1 layer 1')
        makesubplot(fig, nr, nc, 4*nc+c+1, headsl1_2[c,0,:,:], 'transform 2 layer 1')
        makesubplot(fig, nr, nc, 5*nc+c+1, headsl2_1[c,0,:,:], 'transform 1 layer 2')
        makesubplot(fig, nr, nc, 6*nc+c+1, headsl2_2[c,0,:,:], 'transform 2 layer 2')
        makesubplot(fig, nr, nc, 7*nc+c+1, headsl3_1[c,0,:,:], 'transform 1 layer 3')
        makesubplot(fig, nr, nc, 8*nc+c+1, headsl3_2[c,0,:,:], 'transform 2 layer 3')
        makesubplot(fig, nr, nc, 9*nc+c+1, preds1[c,0,:,:], 'transform 1 pred')
        makesubplot(fig, nr, nc, 10*nc+c+1, preds2[c,0,:,:], 'transform 2 pred')
    plt.savefig(savefilename, bbox_inches='tight', dpi=50)
    plt.close()
    return 0

def makesubplot(fig, nr, nc, c, img, title):
    fig.add_subplot(nr, nc, c, xticks=[], yticks=[])
    plt.imshow(img, cmap = 'gray')
    plt.colorbar()
    plt.title(title)
    return 0