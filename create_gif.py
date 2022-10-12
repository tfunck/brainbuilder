import utils.ants_nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sys import argv
from matplotlib import cm
from skimage.transform import resize
import matplotlib.animation as animation
import re

def update(i):
    if dim == 0 :
        rec_frame = rec_vol[i,:,:]
        mri_frame = mri_vol[i,:,:]
    else :
        rec_frame = rec_vol[:,:,i] 
        mri_frame = mri_vol[:,:,i] 

    #rec_frame = np.rot90(rec_frame)

    im = ax.imshow(mri_frame, cmap='grays')
    im = ax.imshow(rec_frame,cmap='nipy_spectral')

    return ax

def compile_frames(dim, rec_vol, mri_vol) :
    images = []
    print('Dim',dim)
    fig, ax = plt.subplots(figsize=(8,8))
    vmin ,vmax = np.percentile(rec_vol[rec_vol>0], [0.2,99.] )
    for i in range(rec_vol.shape[dim]) :
        print('i=',i/rec_vol.shape[dim])
        if dim == 0 :
            rec_frame = rec_vol[i,:,:]
            mri_frame = mri_vol[i,:,:]
        elif dim ==1 :
            rec_frame = rec_vol[:,i,:] 
            mri_frame = mri_vol[:,i,:] 
        elif dim ==2 :
            rec_frame = rec_vol[:,:,i] 
            mri_frame = mri_vol[:,:,i] 
        rec_frame = np.rot90(rec_frame, -1)
        mri_frame = np.rot90(mri_frame, -1)

        if np.sum(rec_frame) > 100  :
            im=ax.imshow(mri_frame, cmap='gist_gray', animated=True)
            im2=ax.imshow(rec_frame,vmin=vmin, vmax=vmax, cmap='nipy_spectral',alpha=0.3, animated=True)
            #im2=ax.imshow(rec_frame,vmin=vmin, vmax=vmax, cmap='nipy_spectral',alpha=1) #, animated=True)
            #plt.savefig(f'gif/{dim}_{i}.png')
            images.append([im, im2])
            #images.append([im2])
            #plt.clf()
            #plt.cla()
    return images

def create_gif(dim,rec_vol,mri_vol,out_fn):
    
    images = compile_frames(dim, rec_vol, mri_vol) :
    anim = animation.ArtistAnimation(fig, images,  blit=True, interval=150, repeat_delay=1000 )
    print('Saving', out_fn)
    anim.save(out_fn,savefig_kwargs={'facecolor':'black'} ) 
    plt.clf()
    plt.cla()
    del images
    del anim

if __name__ == '__main__' :
    rec_file = argv[1] #'MR1_R_flum_1.0mm_space-mni.nii.gz' 
    mri_file = argv[2] #'mri1_t1_tal.nii'
    mask_file = argv[3] #'mri1_brain_mask.mnc'
    #cls_file='mri1_pve_classify.mnc'

    rec_img = nib.load(rec_file)
    rec_vol = rec_img.get_fdata()

    #mask_img = resample_from_to( nib.load(mask_file), rec_img, order=0)
    #mask_vol = mask_img.get_fdata()

    #cls_img = resample_from_to( nib.load(cls_file), rec_img, order=0)
    #cls_vol = cls_img.get_fdata()

    mri_img =nib.load(mri_file) # resample_from_to( nib.load(mri_file), rec_img )
    mri_vol = mri_img.get_fdata()
    mri_vol = resize(mri_vol, rec_vol.shape)

    use_flip = True
    if use_flip:
        rec_vol = np.flip(rec_vol,axis=(0,1,2))
        mri_vol = np.flip(mri_vol,axis=(0,1,2))

    from skimage.morphology import binary_erosion
    #cls_vol[cls_vol != 3 ] = 0
    #cls_vol[cls_vol == 3 ] = 1
    #cls_vol = binary_erosion(cls_vol)
    #rec_vol *= mask_vol
    #rec_vol[cls_vol==1] = 0
    base_path = re.sub('.nii.gz','',os.path.basename(rec_file))

    #vmin = 27
    #vmax = rec_vol.max() * 1.05

    for dim in [0,1,2] :
        out_fn = f'{base_path}-{dim}.gif'
        create_gif(dim,rec_vol,mri_vol,out_fn)



