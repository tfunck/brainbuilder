import os
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import mesh_io
import pandas as pd
import stripy as stripy
import numpy as np
import vast.surface_tools as surface_tools
import vast.surface_tools as io_mesh
from mesh_io import load_mesh_geometry
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_fill_holes
from nibabel.processing import resample_to_output
from skimage.filters import threshold_otsu, threshold_li
from ants import  from_numpy,  apply_transforms
from ants import image_read, registration
from utils.utils import shell
from vast.surface_volume_mapper import SurfaceVolumeMapper

def load_slabs(slab_fn_list) :
    slabs=[]
    for slab_fn in slab_fn_list :
        slabs.append( nib.load(slab_fn) )
    return slabs

def w2v(i, step, start):
    return np.round( (i-start)/step ).astype(int)

def v2w(i, step, start) :
    return start + i * step

def concat_slabs(slabs, dims, mri_ystart, mri_ystep):
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


def write_mask_and_array( array_src_h5 = 'test_array_src.nii.gz', mask_src_h5 = 'test_mask_src.nii.gz' , clobber=0 ) :
    out_dir = 'interslab/'
    mri_fn = 'srv/mri1_gm_bg_srv.nii.gz'

    #slab_fn_list =  ['musc_space-mni_slab-1.nii.gz', 'musc_space-mni_slab-2.nii.gz','musc_space-mni_slab-3.nii.gz','musc_space-mni_slab-4.nii.gz', 'musc_space-mni_slab-5.nii.gz', 'musc_space-mni_slab-6.nii.gz']
    slab_fn_list =  ['flum_no-interp_space-mni_slab-1.nii.gz', 'flum_no-interp_space-mni_slab-2.nii.gz','flum_no-interp_space-mni_slab-3.nii.gz','flum_no-interp_space-mni_slab-4.nii.gz', 'flum_no-interp_space-mni_slab-5.nii.gz', 'flum_no-interp_space-mni_slab-6.nii.gz']

    if not os.path.exists(out_dir) : os.makedirs(out_dir)

    mri_img = nib.load(mri_fn)
    mri_ystart = mri_img.affine[1,3]
    mri_ystep = mri_img.affine[1,1]

    voxel_width = 30

    if not os.path.exists(array_src_h5) or not os.path.exists(mask_src_h5)  or clobber >= 2 :
        slabs = load_slabs(slab_fn_list)
        rec_aff = slabs[0].affine
        mri = mri_img.get_data()
        x_dim = slabs[0].shape[0]
        y_dim = mri.shape[1] 
        z_dim = slabs[0].shape[2]
        array_src, mask_src = concat_slabs(slabs, [x_dim,y_dim,z_dim], mri_ystart, mri_ystep)

        f_array_src = h5.File(array_src_h5, 'w')
        f_mask_src = h5.File(mask_src_h5, 'w')

        f_array_src.create_dataset("data",(x_dim,y_dim,z_dim), dtype=np.float16)
        f_mask_src.create_dataset("data", (x_dim,y_dim,z_dim) , dtype='bool')

        f_array_src['data'][:]=array_src.astype(np.float16)
        f_mask_src['data'][:] = mask_src.astype(np.int)

        f_mask_src.close()
        f_array_src.close()
        del mask_src
        del array_src
        del f_array_src
        del f_mask_src
    return mri_img.affine

def vol_surf_interp(src, coords, surface_fn='surface_slab_mask.txt', clobber=0 ):
    steps = [ affine[0,0], affine[1,1], affine[2,2]   ]
    starts = [ affine[0,3], affine[1,3], affine[2,3]   ]

    vx=np.array([ w2v(i, steps[0], starts[0]) for i in coords[:,0]]).astype(int)
    vy=np.array([ w2v(i, steps[1], starts[1]) for i in coords[:,1]]).astype(int)
    vz=np.array([ w2v(i, steps[2], starts[2]) for i in coords[:,2]]).astype(int)
    val = src['data'][:][vx, vy, vz].reshape(-1,)
    df = pd.DataFrame({'x':coords[:,0], 'y':coords[:,1],'z':coords[:,2],'vx':vx,'vy':vy,'vz':vz,'val':val})
    return val, df


def write_nii(ar, fn, affine, dtype, clobber):
    if not os.path.exists(fn) or clobber >= 1 :
        nib.Nifti1Image(ar['data'][:].astype(dtype), affine).to_filename(fn)


def get_profiles(surf_mid_fn, surf_wm_fn, surf_gray_fn, n_depths, array_src, profile_fn ):
    surf_mid_dict = load_mesh_geometry(surf_mid_fn)
    surf_gm_dict = load_mesh_geometry(surf_gm_fn)
    surf_wm_dict = load_mesh_geometry(surf_wm_fn)
    mid_coords = surf_mid_dict['coords']
    gm_coords = surf_gm_dict['coords']
    wm_coords = surf_wm_dict['coords']
    sphere = load_mesh_geometry(sphere_obj_fn) 

    print('Convert to sphereical coordinates')
    spherical_coords = surface_tools.spherical_np(sphere['coords'])
    print(mid_coords.shape, gm_coords.shape, wm_coords.shape, spherical_coords.shape)

    d_coords = gm_coords-wm_coords
    n_depths=80
    dt = 1.0/ n_depths
    depth_list = np.arange(0., 1+dt, dt)
    profiles=np.zeros([gm_coords.shape[0], len(depth_list)])
    for i, depth in enumerate(depth_list) :
        print('Depth', depth)
        coords = wm_coords + depth * d_coords
        #TODO each depths needs own mask 
        surface_mask, df_mask = vol_surf_interp(mask_src, coords, surface_mask_fn, clobber=args.clobber)
    
        spherical_coords_src = spherical_coords[ surface_mask, : ]
        lats_src, lons_src = spherical_coords_src[:,1]-np.pi/2, spherical_coords_src[:,2]
        lats, lons = spherical_coords[:,1]-np.pi/2, spherical_coords[:,2]
        mesh = stripy.sTriangulation(lons_src,lats_src)
        
        surface_mask = surface_mask.astype(bool)
        surface_val, df_val = vol_surf_interp(array_src, coords, surface_val_fn, clobber=1)
        df_val['mask'] = df_mask.val
        
        surface_val_src = surface_val[ surface_mask.astype(bool) ]
        surface_val_tgt = surface_val[ ~surface_mask.astype(bool) ]

        interp_val, interp_type = mesh.interpolate(lons,lats, zdata=surface_val[surface_mask], order=1)
        
        profiles[:,i] = interp_val
    pd.DataFrame(profiles).to_csv(profile_fn, index=False,header=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--clobber', dest='clobber', type=int, default=0, help='Clobber results')
    parser.add_argument('--brain', dest='brain', type=str, help='brain')
    parser.add_argument('--hemi', dest='hemi', type=str,  help='Clobber results')
    parser.add_argument('--slab', dest='slab', type=str,  help='Clobber results')
    parser.add_argument('--out-dir', dest='out_dir', type=str,  help='Clobber results')
    args = parser.parse_args()

    out_dir = args.out_dir
    brain = args.brain
    hemi = args.hemi
    slab = args.slab

    array_src_h5 = f'{out_dir}/{brain}_{hemi}_{slab}_array_src.h5'
    mask_src_h5  = f'{out_dir}/{brain}_{hemi}_{slab}_array_mask.h5'
    array_src_nii = f'{out_dir}/{brain}_{hemi}_{slab}_array_src.nii.gz'
    mask_src_nii  = f'{out_dir}/{brain}_{hemi}_{slab}_array_mask.nii.gz'
    profile_fn  = f'{out_dir}/{brain}_{hemi}_{slab}_profiles.csv'
    interp_fn  = f'{out_dir}/{brain}_{hemi}_{slab}_interp.nii.gz'

    #Interpolate at coordinate locations
    n_vertices = 327696
    #n_vertices = 163848
    #n_vertices = 81920
    surf_mid_fn = f'civet/mri1/surfaces/mri1_mid_surface_right_{n_vertices}.obj'
    surf_gm_fn = f'civet/mri1/surfaces/mri1_gray_surface_right_{n_vertices}.obj'
    surf_wm_fn = f'civet/mri1/surfaces/mri1_white_surface_right_{n_vertices}.obj'
    sphere_obj_fn = f'receptor_to_mri/civet/mri1/surfaces/mri1_mid_surface_right_{n_vertices}_sphere.obj'

    affine = write_mask_and_array(array_src_h5, mask_src_h5, clobber=args.clobber)
    steps = [ affine[0,0], affine[1,1], affine[2,2]   ]
    starts = [ affine[0,3], affine[1,3], affine[2,3]   ]


    array_src = h5.File(array_src_h5, 'r+')
    mask_src = h5.File(mask_src_h5, 'r+')

    write_nii( array_src, array_src_nii, affine, np.float32, args.clobber)
    write_nii( mask_src,  mask_src_nii, affine, np.int32, args.clobber)

    print('Get surface mask and surface values')
    surface_mask_fn = f'surface_slab_mask_{n_vertices}.txt'
    surface_val_fn  = f'surface_slab_val_{n_vertices}.txt'
    n_depths=4

    if not os.path.exists(profile_fn) or args.clobber >= 1 :
        get_profiles(surf_mid_fn, surf_wm_fn, surf_gm_fn, n_depths, array_src, profile_fn )

    dimensions = array_src['data'][:].shape
    mask_src.close()
    array_src.close()
    
    profiles = pd.read_csv(profile_fn).values

    mapper = SurfaceVolumeMapper(white_surf=surf_wm_fn, gray_surf=surf_gm_fn, resolution=steps, mask=None,dimensions=dimensions, origin=starts, filename=None, save_in_absence=False )

    print('Map Vector to Block')
    vol_interp = mapper.map_profiles_to_block(profiles)

    print('\tWrite volumetric interpolated values')
    nib.Nifti1Image(vol_interp, affine).to_filename(interp_fn)

