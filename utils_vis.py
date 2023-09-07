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
import imageio
from PIL import Image, ImageDraw, ImageFont
from skimage import data, color
from skimage.segmentation import mark_boundaries


def get_center_roi(label):
    
    # Get row and column indices of non-zero elements
    rows, cols = np.nonzero(label)

    # Calculate mean row and column indices
    mean_row = np.mean(rows)
    mean_col = np.mean(cols)

    # Round to nearest integer to get pixel coordinates
    return int(np.round(mean_row)), int(np.round(mean_col))

def resolve_boundary(c, n, d):
    if c < d:
        s = 0
        e = 2 * d
    elif c > n - d:
        s = - 2 * d
        e = -1
    else:
        s = c - d
        e = c + d
    
    return s, e

# ===================
# visualize results
# ===================
def save_results_zoom(image, # x, y, nz
                      label, # x, y, nz
                      logits, # nz, nc, x, y
                      soft_pred, # nz, nc, x, y
                      hard_pred, # nz, x, y
                      savepath):
    
    # find slice with largest foreground
    zz = np.argmax(np.sum(label, axis=(0,1)))
    # find center of foreground in this slice
    lbl_tmp = np.copy(label[:,:,zz])
    # roi is centered around this pixel
    cx, cy = get_center_roi(lbl_tmp)
    nx, ny = label.shape[0], label.shape[1]
    d = 60
    sx, ex = resolve_boundary(cx, nx, d)
    sy, ey = resolve_boundary(cy, ny, d)

    # 
    img = image[sx:ex, sy:ey, zz]
    lbl = label[sx:ex, sy:ey, zz]
    img_lbl = mark_boundaries(img, lbl, color=(0,0,1), mode='thick')
    prd = hard_pred[zz, sx:ex, sy:ey]
    img_prd = mark_boundaries(img, prd, color=(0,1,0), mode='thick')
    logs = logits[zz, :, sx:ex, sy:ey]
    probs = soft_pred[zz, :, sx:ex, sy:ey]

    save_image_wo_normalization(img, savepath + '_img.png')
    # save_image_wo_normalization(lbl, savepath + '_lbl.png')
    save_image_wo_normalization(img_lbl, savepath + '_lbl.png', None)
    save_image_wo_normalization(img_prd, savepath + '_prd.png', None)
    for c in range(1, logs.shape[0]):
        save_image_wo_normalization(logs[c,:,:], savepath + '_logits' + str(c) + '.png', 'jet')
        save_image_wo_normalization(probs[c,:,:], savepath + '_probs' + str(c) + '.png', 'jet')

    return 0

# ==========================================================
# ==========================================================
def save_image_wo_normalization(image,
                                savepath,
                                cmap = 'gray'):
    
    plt.figure(figsize=(5, 5))
    if cmap != None:
        plt.imshow(image, cmap = cmap)
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()
    return 0

# ==========================================================
# ==========================================================
def save_image(image, savepath, cmap = 'gray'):
    
    plt.figure(figsize=(5, 5))
    k=0
    plt.imshow(np.rot90(normalize_img_for_vis(image), k), cmap = cmap)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()
    return 0

# ==========================================================
# ==========================================================
def save_images_labels_all(images,
                           labels,
                           savepath):
    
    bs = images.shape[0]
    plt.figure(figsize=(4*2, 4*bs))
    k=0
    s=50
    e=200
    # show 4 examples per batch
    for batch_index in range(bs):
        plt.subplot(bs, 2, 2*batch_index + 1, xticks=[], yticks=[])
        plt.imshow(np.rot90(normalize_img_for_vis(images[batch_index,0,s:e,s:e]),k), cmap = 'gray')
        plt.subplot(bs, 2, 2*batch_index + 2, xticks=[], yticks=[])
        plt.imshow(np.rot90(normalize_img_for_vis(labels[batch_index,0,s:e,s:e]),k), cmap = 'gray')
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()
    return 0

# ==========================================================
# ==========================================================
def save_gif(image,
             ground,
             preds,
             dices,
             savepath,
             s=75,
             e=-75):
    
    images_vis = [np.random.randint(0, 10, size=(170, 170, 3), dtype=np.uint8) for i in range(len(preds))]
    
    for idx in range(len(images_vis)):
        images_vis[idx] = normalize_img_for_vis(mark_boundaries(image[s:e,s:e], preds[idx][s:e,s:e], color=(1, 1, 1), mode='thick'))

    # Convert the numpy arrays to PIL images
    pil_images = [Image.fromarray(img) for img in images_vis]

    # Add a title to each frame
    for i, pil_image in enumerate(pil_images):
        draw = ImageDraw.Draw(pil_image)
        draw.text((10, 10), "Dice {}".format(dices[i]), (255, 0, 0))

    # Save the list as a gif
    pil_images[0].save(savepath,
                       save_all=True,
                       append_images=pil_images[1:],
                       duration=100,
                       loop=1)
    
# ==========================================================
# ==========================================================
def save_all(all,
             savepath,
             torch_or_numpy = 'torch',
             s=50,
             e=200,
             cmaps=None):
    
    num_things = len(all)
    bs = all[0].shape[0]
    
    plt.figure(figsize=(4*num_things, 4*bs))
    k=0
    
    for batch_index in range(bs):

        for t in range(num_things):

            if cmaps != None:
                cmap = cmaps[t]
            else:
                cmap = 'gray'
            
            plt.subplot(bs, num_things, num_things*batch_index + t + 1, xticks=[], yticks=[])
            if torch_or_numpy == 'torch':
                plt.imshow(np.rot90(all[t].detach().cpu().numpy()[batch_index,s:e,s:e], k), cmap = cmap)
            else:
                plt.imshow(np.rot90(all[t][batch_index,s:e,s:e], k), cmap = cmap)
            plt.colorbar()
    
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()
    
    return 0

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
    plt.close()
    
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
    plt.close()

    return 0

# ==========================================================
# ==========================================================
def show_images_labels_preds(images,
                             labels, # hard labels (not 1 hot)
                             preds): # soft predictions
    
    fig = plt.figure(figsize=(18, 24))
    
    tmp_images = np.copy(images.cpu().numpy())
    tmp_labels = np.copy(labels.cpu().numpy())
    tmp_preds = np.copy(preds.detach().cpu().numpy())

    tmp_images = np.squeeze(tmp_images)
    tmp_labels = np.squeeze(tmp_labels)
    tmp_preds = np.argmax(tmp_preds, axis = 1)

    # show 4 examples per batch
    for batch_index in range(4):
        ax = fig.add_subplot(4, 3, 3*batch_index + 1, xticks=[], yticks=[])
        plt.imshow(normalize_img_for_vis(tmp_images[batch_index,:,:]), cmap = 'gray')
        ax = fig.add_subplot(4, 3, 3*batch_index + 2, xticks=[], yticks=[])
        plt.imshow(normalize_img_for_vis(tmp_labels[batch_index,:,:]), cmap = 'gray')
        ax = fig.add_subplot(4, 3, 3*batch_index + 3, xticks=[], yticks=[])
        plt.imshow(normalize_img_for_vis(tmp_preds[batch_index,:,:]), cmap = 'gray')

    return fig

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
def get_title(layer_num):
    if layer_num == 0:
        t = 'Input'
    elif layer_num == 11:
        t = 'Output'
    elif layer_num == 10:
        t = 'Layer L'
    else:
        t = 'Layer L-' + str(10 - layer_num)
    return t

# clims for variance
def get_clim(layer_num):
    if layer_num == 0:
        c = [0.0, 0.05]
    elif layer_num == 11:
        c = [0.0, 0.25]
    elif layer_num == 10:
        c = [0.0, 300.0]
    else:
        c = [0.0, 1.0]
    return c

# ==========================================================
# ==========================================================
def show_stats(stats,
               vis_layers,
               savepath):
    nr = 1
    nc = len(vis_layers)
    plt.figure(figsize=(4*nc, 4*nr))
    for l in range(len(vis_layers)):
        stat_l = stats['layer' + str(vis_layers[l])]
        plt.subplot(nr, nc, 0*nc + l + 1)
        plt.axis('off')
        plt.imshow(stat_l, cmap = 'gray')
        # plt.clim(get_clim(vis_layers[l]))
        plt.colorbar()
        plt.title(get_title(vis_layers[l]) + ' ,' + str(np.round(np.mean(stat_l),2)))
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()
    
# ==========================================================
# ==========================================================
def show_prediction_variation_2(means,
                                vars,
                                savepath):

    # show_stats(means, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], savepath + '_means_all_layers.png')
    # show_stats(vars, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], savepath + '_variances_all_layers.png')
    show_stats(means, [0, 6, 7, 8, 9, 10, 11], savepath + '_means_decoder.png')
    show_stats(vars, [0, 6, 7, 8, 9, 10, 11], savepath + '_variances_decoder.png')

    return 0

# ==========================================================
# ==========================================================
def show_prediction_variation(images,
                              labels,
                              pred_mean,
                              pred_stddev,
                              savepath):
    
    plt.figure(figsize=(24, 24))

    # show 4 examples per batch
    for batch_index in range(4):
        plt.subplot(4, 4, 4*batch_index + 1); plt.imshow(images[batch_index,0,:,:], cmap = 'gray'); plt.colorbar()
        plt.subplot(4, 4, 4*batch_index + 2); plt.imshow(labels[batch_index,0,:,:], cmap = 'gray'); plt.colorbar()
        plt.subplot(4, 4, 4*batch_index + 3); plt.imshow(pred_mean[batch_index,0,:,:], cmap = 'gray'); plt.colorbar()
        plt.subplot(4, 4, 4*batch_index + 4); plt.imshow(pred_stddev[batch_index,0,:,:], cmap = 'gray'); plt.colorbar()
    
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()

    return 0

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
def save_from_list(list_of_things,
                   savefilename,
                   normalize = False,
                   rot90_k = -1):
    
    num_things = len(list_of_things)
    tmp_list = []

    for thing in list_of_things:
        tmp_list.append(thing.detach().cpu().numpy())

    plt.figure(figsize=(6*num_things, 24))

    # show 4 examples per batch
    for batch_index in range(4):
        i = 1
        for thing in tmp_list:
            if normalize == True:
                im = normalize_img_for_vis(thing[batch_index,0,:,:])
            else:
                im = thing[batch_index,0,:,:]
            if rot90_k != 0:
                im = np.rot90(im, k=rot90_k)
            plt.subplot(4, num_things, num_things*batch_index + i, xticks=[], yticks=[]); plt.imshow(im, cmap = 'gray'); plt.colorbar()
            i = i + 1
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

def plot_scatter(r, savepath):
    plt.figure(figsize=(10, 10))
    plt.scatter(np.mean(r, 0), np.std(r, 0))
    plt.xlabel('Mean Dice over 500 successive training iterations (step size 1e-8)')
    plt.ylabel('Std deviation Dice over 500 successive training iterations (step size 1e-8)')
    plt.ylim([-0.005, 0.07])
    plt.title('Each dot is one test subject')
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()
    return 0

def plot_subjectwise(r, savepath):
    plt.figure(figsize=(10, 10))
    for s in range(r.shape[-1]):
        plt.plot(r[:,s])
    plt.title('Each line is one test subject. Mean ' + str(np.round(np.mean(r[:,:]), 2)))
    plt.ylim([-0.05, 1.0])
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()
    return 0

def plot_methods(res, leg, savepath):
    plt.figure(figsize=(30, 10))
    for rid in range(len(res)):
        plt.plot(res[rid], label=leg[rid])
    # plt.ylim([-0.05, 1.0])
    plt.legend()
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()
    return 0

def plot_scatter_simple(a,
                        b,
                        c,
                        savepath,
                        measure_type):
    
    numruns = len(a)
    colors = ['blue', 'green', 'red']

    plt.figure(figsize=(10, 10))
    corr = 0.0
    for r in range(numruns):
        a1 = np.mean(a[r],-1) # mean across subjects
        b1 = np.mean(b[r],-1) # mean across subjects
        corr = corr + np.corrcoef(a1, b1)[0,1]
        plt.scatter(a1, b1, color=colors[r])
        # highlight iteration with highset val score
        ind = np.argmax(a1)
        plt.scatter(a1[ind], b1[ind],  c=colors[r],  marker='*', s=200)
        # highlight last iteration
        # plt.scatter(a1[-1], b1[-1],  c=colors[r],  marker='d', s=200)

    plt.xlabel('Val InD', fontsize=22)
    plt.ylabel('Test InD', fontsize=22)
    plt.ylim([0.5, 1.0])
    plt.title('Corr: ' + str(np.round(corr / numruns, 2)), fontsize=22)
    plt.savefig(savepath + '/corr_ind_' + measure_type + '.png', bbox_inches='tight', dpi=50)
    plt.close()

    plt.figure(figsize=(10, 10))
    corr = 0.0
    for r in range(numruns):
        a1 = np.mean(a[r],-1) # mean across subjects
        c1 = np.mean(c[r],-1) # mean across subjects
        corr = corr + np.corrcoef(a1, c1)[0,1]
        plt.scatter(a1, c1, color=colors[r])
        # highlight iteration with highset val score
        ind = np.argmax(a1)
        plt.scatter(a1[ind], c1[ind],  c=colors[r],  marker='*', s=200)
        # highlight last iteration
        # plt.scatter(a1[-1], c1[-1],  c=colors[r],  marker='d', s=200)

    plt.xlabel('Val InD', fontsize=22)
    plt.ylabel('Test OoD', fontsize=22)
    plt.ylim([0.5, 0.8])
    plt.title('Corr: ' + str(np.round(corr / numruns, 2)), fontsize=22)
    plt.savefig(savepath + '/corr_ood_' + measure_type + '.png', bbox_inches='tight', dpi=50)
    plt.close()

    return 0

# ============================================================================
# ============================================================================
def plot_ind_ood_corr_sub_dataset(a1, a2, savepath, dataset):

    # ====================================
    # plots of dice evolution over training iters
    # ====================================
    iters = np.linspace(10000, 100000, 10)
    
    plt.figure(figsize=(3, 8))
    
    # plot evolution of each val subject
    for s in range(a1.shape[-1]):
        plt.plot(iters, a1[:,s], 'r', alpha=0.25)
    plt.plot(iters, np.mean(a1, -1), 'r', linewidth=2)

    # plot evolution of each test subject
    for s in range(a2.shape[-1]):
        plt.plot(iters, a2[:,s], 'b', alpha=0.25)
    plt.plot(iters, np.mean(a2, -1), 'b', linewidth=2)

    plt.xlabel('Training iterations', fontsize=12)
    plt.ylabel('Dice', fontsize=12)
    plt.ylim([0.5, 0.95])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(savepath + '_' + dataset + '_plot.png', bbox_inches='tight', dpi=100)
    plt.close()

    # ====================================
    # scatter of val dice vs test dice
    # ====================================
    plt.figure(figsize=(3, 3))
    a11 = np.mean(a1,-1)
    a22 = np.mean(a2,-1)
    plt.scatter(a11, a22)
    corr = np.corrcoef(a11[1:], a22[1:])[0,1] # ignore values at 10k to see correlation once val becomes constant
    plt.xlabel('InD Val', fontsize=12)
    if dataset == 'BIDMC':
        plt.ylim([0.55, 0.75])
        plt.ylabel('OoD Test', fontsize=12)
    elif dataset == 'RUNMC':
        plt.ylim([0.85, 0.95])
        plt.ylabel('InD Test', fontsize=12)
    else:
        plt.ylim([0.70, 0.85])
        plt.ylabel('OoD Test', fontsize=12)
    plt.xlim([0.89, 0.905])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(np.round(corr, 2))
    plt.savefig(savepath + '_' + dataset + '_scatter.png', bbox_inches='tight', dpi=100)
    plt.close()

# ============================================================================
# a1 [num_runs, num_models, num_subjects]
# ============================================================================
def plot_ind_ood_corr_2_sub_dataset(a1, a2, filename):

    iters = np.linspace(10000, 100000, 10)
    
    plt.figure(figsize=(3, 3))
    
    # for each run
    corr = 0.0
    for r in range(a1.shape[0]):
        a11 = np.mean(a1[r,:,:],-1)
        a22 = np.mean(a2[r,:,:],-1)
        plt.scatter(a11, a22)
        corr = corr + np.corrcoef(a11,a22)[0,1]

    plt.xlabel('InD Val', fontsize=12)
    
    if 'BIDMC' in filename[-15:]:
        plt.ylim([0.55, 0.75])
        plt.ylabel('OoD Test', fontsize=12)
    elif 'RUNMC' in filename[-15:]:
        plt.ylim([0.85, 0.95])
        plt.ylabel('InD Test', fontsize=12)
    else:
        plt.ylim([0.70, 0.85])
        plt.ylabel('OoD Test', fontsize=12)
    plt.xlim([0.89, 0.905])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(np.round(corr/3, 2))
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()

# ============================================================================
# ============================================================================
def plot_ind_ood_corr(ind_val,
                      ind_tst,
                      ood_tst,
                      savepath):
    
    plot_ind_ood_corr_sub_dataset(ind_val, ind_tst, savepath, 'RUNMC')
    plot_ind_ood_corr_sub_dataset(ind_val, ood_tst[:,:10], savepath, 'BMC')
    plot_ind_ood_corr_sub_dataset(ind_val, ood_tst[:,10:20], savepath, 'UCL')
    plot_ind_ood_corr_sub_dataset(ind_val, ood_tst[:,20:30], savepath, 'HK')
    plot_ind_ood_corr_sub_dataset(ind_val, ood_tst[:,30:40], savepath, 'BIDMC')

# ============================================================================
# ============================================================================
def plot_ind_ood_corr2(ind_val,
                       ind_tst,
                       ood_tst,
                       savepath):

    plot_ind_ood_corr_2_sub_dataset(ind_val, ind_tst, savepath + '_RUNMC.png')
    plot_ind_ood_corr_2_sub_dataset(ind_val, ood_tst[:,:,:10], savepath + '_BMC.png')
    plot_ind_ood_corr_2_sub_dataset(ind_val, ood_tst[:,:,10:20], savepath + '_UCL.png')
    plot_ind_ood_corr_2_sub_dataset(ind_val, ood_tst[:,:,20:30], savepath + '_HK.png')
    plot_ind_ood_corr_2_sub_dataset(ind_val, ood_tst[:,:,30:40], savepath + '_BIDMC.png')

# ============================================================================
# ============================================================================
def find_roi(arr,
             roi = 'full', # full / corner
             delta = 15): 
    
    idx = np.sum(arr, axis = 1)
    idxx = np.nonzero(idx)[0]
    sx = idxx[0] - delta
    if roi == 'full':
        ex = idxx[-1] + delta
    elif roi == 'corner':
        ex = idxx[0] + delta
    
    idy = np.sum(arr, axis = 0)
    idyy = np.nonzero(idy)[0]
    sy = idyy[0] - delta
    if roi == 'full':
        ey = idyy[-1] + delta
    elif roi == 'corner':
        ey = idyy[0] + delta
    
    return sx, ex, sy, ey

# ============================================================================
# ============================================================================
def add_subplot(ax,
                fig,
                arr,
                title,
                sx = 60,
                ex = -100,
                sy = 60,
                ey = -60,
                cmap = 'gray',
                colorbar = 'on'):
    
    im = ax.imshow(arr[sx:ex, sy:ey], cmap = cmap)
    if colorbar == 'on':
        fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return 0

# ================================================
# ================================================
def show_array(arr, title = ''):
    plt.figure()
    plt.imshow(arr, 'gray')
    plt.colorbar()
    plt.title(title)
    plt.show()
    plt.close()