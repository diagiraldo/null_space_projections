# Diana Giraldo
# Jan, 2023

import matplotlib.pyplot as plt
import numpy as np
    
def show_3dpatch_ortho(
    image,
    cross = None,
    show_cross = False,
    orient = "row", # or col
    patchsize = 2, 
    vmin = 0., 
    vmax = 1.,
    cmap = "gray",
    out_file = None,
    overlay_image = None,
    overlay_vmax = 1.,
    overlay_cmap = None,
    overlay_alpha = None,
    return_fig = False,
):
    
    if cross is None:
        cross = np.rint((np.array(image.shape)-1)/2).astype(np.int32)
        
    patch_list = [
        image[:, cross[1], :],
        image[cross[0], :, :],  
        image[:, :, cross[2]]
    ]
    
    if show_cross:
        lines = [
            (cross[0], cross[2]),
            (cross[1], cross[2]),
            (cross[0], cross[1])
        ]
        
    if overlay_image is not None:
        ov_list = [
            overlay_image[:, cross[1], :],
            overlay_image[cross[0], :, :],  
            overlay_image[:, :, cross[2]]
        ]
    
    if orient == "row":
        nrows = 1
        ncols = 3
    elif orient == "col":
        nrows = 3
        ncols = 1
    else:
        raise ValueError('orientation not valid')
    
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, 
                             figsize = (ncols*patchsize,nrows*patchsize),
                             squeeze = False)
       
    for i in range(nrows):
        for j in range(ncols):
            patch = patch_list[i*ncols + j]
            axes[i,j].imshow(patch.swapaxes(1,0), cmap = cmap, origin = "lower", interpolation = "none", aspect = "auto", vmin = vmin, vmax = vmax)
            axes[i,j].axis("off")
            
            if overlay_image is not None:
                ov_img = ov_list[i*ncols + j].swapaxes(1,0)
                alpha_mask = overlay_alpha*ov_list[i*ncols + j].swapaxes(1,0)
                if len(alpha_mask.shape) == 3:
                    alpha_mask = np.max(alpha_mask, axis = 2, keepdims=True)
                    ov_img = np.concatenate((ov_img, alpha_mask), axis = 2)
                    alpha_mask = None
                    
                axes[i,j].imshow(
                    ov_img, 
                    cmap = overlay_cmap, 
                    alpha = alpha_mask,
                    origin = "lower", 
                    interpolation = "nearest", 
                    aspect = "auto",
                    vmin = 0, vmax = overlay_vmax
                )
            
            if show_cross:
                axes[i,j].axvline(x = lines[i*ncols + j][0], c = "white", linestyle='--', lw = 0.5)
                axes[i,j].axhline(y = lines[i*ncols + j][1], c = "white", linestyle='--', lw = 0.5)
            
    fig.tight_layout()
    
    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)

    if return_fig:
        return fig,axes
    else:
        plt.show()
        plt.close()
        
def show_orthoslices(image, 
                     cross = None, 
                     figsize = (18,6), 
                     cmap = "gray", 
                     vmin = None, 
                     vmax = None, 
                     show_cross = False,
                     out_file = None):
    
    slices = [image[cross[0], :, :], image[:, cross[1], :], image[:, :, cross[2]]]
    dims = [("Dimension 2", "Dimension 3"), ("Dimension 1", "Dimension 3"), ("Dimension 1", "Dimension 2")]

    if vmin is None:
        vmin = np.min(image)
        
    if vmax is None:
        vmax = np.max(image)
        
    if show_cross:
        lines = [
            (cross[1], cross[2]),
            (cross[0], cross[2]),
            (cross[0], cross[1])
        ]
    
    # create plots and save image
    fig, axes = plt.subplots(1, len(slices), figsize = figsize)
    ims = []
    for i, slc in enumerate(slices):
        # subplots, labels and colorbars
        im = axes[i].imshow(slc.T, cmap = cmap, origin = "lower", interpolation = "none", vmin = vmin, vmax = vmax)
        axes[i].set_xlabel(dims[i][0])
        axes[i].set_ylabel(dims[i][1])
        axes[i].set_aspect(slc.shape[0]/slc.shape[1])
        axes[i].set_xticks([]) 
        axes[i].set_yticks([])
        
        if show_cross:
            axes[i].axvline(x = lines[i][0], c = "white", linestyle='--', lw = 0.5)
            axes[i].axhline(y = lines[i][1], c = "white", linestyle='--', lw = 0.5)
        
        ims.append(im)

    left = axes[0].get_position().x0
    bottom = axes[0].get_position().y1 + 0.05
    width = abs(axes[0].get_position().x0 - axes[2].get_position().x1)
    height = 0.02
    cax = fig.add_axes([left, bottom, width, height])
    fig.colorbar(ims[0], cax=cax, orientation="horizontal")
    # fig.set_facecolor("w")
    
    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
    
    plt.show()
    plt.close()