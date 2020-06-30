import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_fill_holes
from nibabel.processing import resample_to_output
from skimage.filters import threshold_otsu, threshold_li
from ants import  from_numpy,  apply_transforms
from ants import image_read, registration



def load_slabs(slab_fn_list) :
    slabs=[]
    for slab_fn in slab_fn_list :
        slabs.append( nib.load(slab_fn) )
    return slabs

def w2v(i, step, start):
    return np.round( (i-start)/step ).astype(int)
def v2w(i, step, start) :
    return start + i * step

def get_tfm(out_dir, fixed, moving, from_position, to_position,clobber=False):
    prefix='%sfrom-%s_to-%s_'%(out_dir, from_position, to_position)
    mat_fn='%s0GenericAffine.mat'%prefix

    if not os.path.exists(mat_fn) or clobber :
        reg = registration(fixed=from_numpy(fixed), moving=from_numpy(moving), type_of_transform='Affine', reg_iterations=(100,70,10), aff_metric='GC', syn_metic='CC', outprefix=prefix  )
    return mat_fn


def get_missing_voxels(slabs, mask_src, mask_tgt, mri, dims, voxel_width=15):
    nslabs=len(slabs)
    for i in range(nslabs) :
        print('Figuring out what are the voxels that need to be interpolated for slab',i)
        j = i + 1
        if i < nslabs - 1 : 
            ant_ydim = slabs[i].shape[1]
            pos_ydim = slabs[j].shape[1]
            ant_ystart = slabs[i].affine[1,3]
            ant_ystep = slabs[i].affine[1,1]

            ant_end = w2v( v2w(voxel_width, ant_ystep, ant_ystart), mri_ystep , mri_ystart )
            ant_start = w2v( v2w(0,ant_ystep, ant_ystart), mri_ystep , mri_ystart )

            pos_ystart = slabs[j].affine[1,3]
            pos_ystep = slabs[j].affine[1,1]

            pos_start = w2v( v2w( pos_ydim-voxel_width, pos_ystep, pos_ystart), mri_ystep , mri_ystart ) 
            pos_end = w2v( v2w(pos_ydim, pos_ystep, pos_ystart), mri_ystep , mri_ystart )
           
            mask_tgt_temp = mri[:,pos_start:ant_end,:]
            mask_src_temp = mask_src['data'][:,int(pos_start):int(ant_end),:]
            temp = np.zeros_like(mask_src_temp)
            temp[ (mask_src_temp == 0) & (mask_tgt_temp == 1 ) ] = 1
            mask_tgt['data'][:,int(pos_start):int(ant_end),:] = temp
    return mask_tgt

def get_kernel_dim(A,dim) :
    A0 = map(lambda i: max(0,i-1), A)
    A1 = map(lambda i: min(dim,i+2), A)
    return A0, A1

def interp_y_axis(array, mask_src,mask_tgt, affine):
    dims=mask_tgt['data'].shape

    free_voxels = np.sum(mask_tgt['data'][:].astype(int))
    old_free_voxels = free_voxels+1
    while free_voxels > 0 and old_free_voxels > free_voxels:
        border = binary_dilation(mask_src['data'][:],iterations=1)
        border *= mask_tgt['data'][:]

        print(f'Free voxels: {free_voxels}')
        x_visited=np.array([])
        y_visited=np.array([])
        z_visited=np.array([])
        for x in range(dims[0]) :
            perc = float(100. * x / dims[0] )
            print(f'\t{perc:3}\r',end="")
            if np.sum(border[x,:,:]) == 0 : continue

            for y in range(dims[1]) :
                if np.sum(border[x,y,:]) == 0 : continue

                for z in range(dims[2]) :
                    if border[x,y,z] :
                        x0=max(0, x-1)
                        x1=min(dims[0], x+2)
                        y0=max(0, y-1)
                        y1=min(dims[1], y+2)
                        z0=max(0, z-1)
                        z1=min(dims[2], z+2)

                        temp_array_src = array['data'][x0:x1,y0:y1,z0:z1]
                        temp_mask = mask_src['data'][x0:x1,y0:y1,z0:z1]
                        if np.sum(temp_mask) > 0 :
                            ii = np.mean(temp_array_src[temp_mask])
                            array['data'][int(x),int(y),int(z)] = ii
                            x_visited=np.append(x_visited,x)
                            y_visited=np.append(y_visited,y)
                            z_visited=np.append(z_visited,z)

        for x, y ,z in zip(x_visited, y_visited, z_visited) :
            mask_tgt['data'][ x,y,z ] = False
            mask_src['data'][ x,y,z ] = True
        print()
        old_free_voxels = free_voxels
        free_voxels = np.sum(mask_tgt['data'][:].astype(int))
        del border

    nib.Nifti1Image(array['data'][:].astype(np.float32), mri_img.affine).to_filename('test_interslab_interp.nii.gz')
    

def concat_slabs(slabs, dims):
    out_mask =np.zeros(dims).astype(bool)
    out =np.zeros(dims)
    nslabs = len(slabs)
    tfm_dict={}
    for i in range(nslabs) :
        print('Anterior Slab:', i, nslabs)
        j = i + 1

        #  pos (n) <-  ant (0)
        #    ____     ____     ____
        #   |    |   |    |   |    |
        #   | S2 |   | S1 |   | S0 |
        #   |____|   |____|   |____|
        #       ||   ||  ||   || 
        #        -> <-    -> <-                

        rec_ant = slabs[i].get_data() 

        rec_ant_mask = np.zeros_like(rec_ant)
        idx = rec_ant > threshold_otsu(rec_ant)
        rec_ant_mask[ idx ] = 1

        rec_ant = (rec_ant - np.mean(rec_ant))/np.std(rec_ant)

        rec_ant_y_start = w2v( v2w( 0 ,slabs[i].affine[1,1], slabs[i].affine[1,3]),  mri_ystep , mri_ystart )
        out_mask[:,rec_ant_y_start:(rec_ant_y_start+rec_ant.shape[1]),:] = rec_ant_mask
        out[:,rec_ant_y_start:(rec_ant_y_start+rec_ant.shape[1]),:] = rec_ant
    return out, out_mask

out_dir = 'interslab/'
mri_fn = 'srv/mri1_gm_bg_srv.nii.gz'

slab_fn_list =  ['musc_space-mni_slab-1.nii.gz', 'musc_space-mni_slab-2.nii.gz','musc_space-mni_slab-3.nii.gz','musc_space-mni_slab-4.nii.gz', 'musc_space-mni_slab-5.nii.gz', 'musc_space-mni_slab-6.nii.gz']
#slab_fn_list =  ['musc_space-mni_slab-5.nii.gz', 'musc_space-mni_slab-6.nii.gz']

if not os.path.exists(out_dir) : os.makedirs(out_dir)

slabs = load_slabs(slab_fn_list)
mri_img = nib.load(mri_fn)
rec_aff = slabs[0].affine
mri = mri_img.get_data()
mri_ystart = mri_img.affine[1,3]
mri_ystep = mri_img.affine[1,1]

voxel_width = 30
#y_all = np.arange(mri.shape[1]).astype(int)
#y_mask = np.sum(mri, axis=(0,2))

x_dim = slabs[0].shape[0]
y_dim = mri.shape[1] 
z_dim = slabs[0].shape[2]

array_src_fn = 'test_array_src.nii.gz'
mask_src_fn = 'test_mask_src.nii.gz'
mask_tgt_fn = 'test_mask_tgt.nii.gz'

clobber=True
if not os.path.exists(array_src_fn) or not os.path.exists(mask_src_fn)  or clobber  :
    array_src, mask_src = concat_slabs(slabs, [x_dim,y_dim,z_dim])

    f_array_src = h5.File(array_src_fn, 'w')
    f_mask_src = h5.File(mask_src_fn, 'w')

    f_array_src.create_dataset("data",(x_dim,y_dim,z_dim), dtype=np.float16)
    f_mask_src.create_dataset("data", (x_dim,y_dim,z_dim) , dtype='bool')

    f_array_src['data'][:]=array_src.astype(np.float16)
    f_mask_src['data'][:] = mask_src.astype(np.int)
    del mask_src
    del array_src
    del f_array_src
    del f_mask_src

array_src = h5.File(array_src_fn, 'r+')
mask_src = h5.File(mask_src_fn, 'r+')

if not os.path.exists(mask_tgt_fn) or clobber :
    f_mask_tgt = h5.File(mask_tgt_fn, 'w')
    f_mask_tgt.create_dataset("data", (x_dim,y_dim,z_dim) , dtype='bool')
    get_missing_voxels(slabs,  mask_src,f_mask_tgt, mri, [x_dim,y_dim,z_dim], voxel_width=voxel_width)
    del f_mask_tgt
del mri
mask_tgt = h5.File(mask_tgt_fn, 'r+')


interp_y_axis(array_src, mask_src, mask_tgt, mri_img.affine)

#nib.Nifti1Image(mask_src['data'].astype(np.int16), mri_img.affine).to_filename('test_final_mask_src.nii.gz')
#nib.Nifti1Image(array_src['data'][:].astype(np.float32), mri_img.affine).to_filename('test_interslab_interp.nii.gz')

#array_src[ mask_src == 1 ] = out_interp[ mask_src == 1 ]
#nib.Nifti1Image(array_src, mri_img.affine).to_filename('test_intraslab.nii.gz')



