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
from apply_ants_transform_to_obj import apply_ants_transform_to_obj
from re import sub
from glob import glob
from mesh_io import load_mesh_geometry, save_mesh_data, save_obj, read_obj
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_fill_holes
from nibabel.processing import resample_to_output
from skimage.filters import threshold_otsu, threshold_li
from ants import  from_numpy,  apply_transforms, apply_ants_transform, read_transform
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

def vol_surf_interp(val, src, coords, affine, surface_fn='surface_slab_mask.txt', clobber=0 ):

    steps = [0.2,0.2,0.2] # [ affine[0,0], affine[1,1], affine[2,2]   ]
    starts =[-90, -126, -72] # [ affine[0,3], affine[1,3], affine[2,3]   ]

    vx=np.array([ w2v(i, steps[0], starts[0]) for i in coords[:,0]]).astype(int)
    vy=np.array([ w2v(i, steps[1], starts[1]) for i in coords[:,1]]).astype(int)
    vz=np.array([ w2v(i, steps[2], starts[2]) for i in coords[:,2]]).astype(int)

    print(affine)
    idx = (vx >= 0) & (vy >= 0) & (vz >= 0) & (vx < src.shape[0]) & ( vy < src.shape[1]) & ( vz < src.shape[2] )
    print('idx',np.sum(idx)) 
    if np.sum(idx) == 0 : exit(1)

    val=np.zeros(vx.shape)
    val[idx] = src[vx[idx], vy[idx], vz[idx]].reshape(-1,)
    df = pd.DataFrame({'x':coords[:,0], 'y':coords[:,1],'z':coords[:,2],'vx':vx,'vy':vy,'vz':vz,'val':val})
    return val, df


def write_nii(ar, fn, affine, dtype, clobber):
    if not os.path.exists(fn) or clobber >= 1 :
        nib.Nifti1Image(ar['data'][:].astype(dtype), affine).to_filename(fn)


def get_profiles(surf_mid_list, surf_wm_list, surf_gray_list, n_depths,  profile_fn, slab_list ):
    dt = 1.0/ n_depths
    depth_list = np.arange(0., 1+dt, dt)
    
    nrows = pd.read_csv(surf_mid_list[0]).shape[0]

    profiles=np.zeros([nrows, len(depth_list)])

    surface_mask=np.zeros(nrows)
    surface_val=np.zeros(nrows)

    nslabs=len(slab_list)
    for slab in range(1, 1+nslabs) :
        i = slab - 1
        print('surf wm fn',surf_wm_list[i])
        print('surf gm fn',surf_gm_list[i])
        print('slab fn', slab_list[i])
        surf_mid_dict = pd.read_csv(surf_mid_list[i])
        surf_gm_dict =  pd.read_csv(surf_gm_list[i])
        surf_wm_dict =  pd.read_csv(surf_wm_list[i])
        
        gm_coords = surf_gm_dict.loc[ :, ['x','y','z'] ].values
        wm_coords = surf_wm_dict.loc[ :, ['x','y','z'] ].values
        
        d_coords = gm_coords - wm_coords
        array_img = nib.load(slab_list[i])
        array_src = array_img.get_fdata()
        for i, depth in enumerate(depth_list) :
            #
            coords = wm_coords + depth * d_coords
            #
            temp_surface_val, df_val = vol_surf_interp(surface_val, array_src, coords, array_img.affine, surface_val_fn, clobber=1)
            surface_val += temp_surface_val
            print('Depth', depth, np.sum(surface_val))
        
    print('Convert to sphereical coordinates')
    #load sphere mesh
    sphere = load_mesh_geometry(sphere_obj_fn) 
    # get coordinates from dicitonary with mesh info
    spherical_coords = surface_tools.spherical_np(sphere['coords'])
   
    #define a mask of verticies where we have receptor densitiies
    surface_mask = surface_val != 0 

    #define vector with receptor densities 
    surface_val_src = surface_val[ surface_mask.astype(bool) ]
    
    #define vector without receptor densities
    surface_val_tgt = surface_val[ ~surface_mask.astype(bool) ]

    # get coordinates for vertices in mask
    spherical_coords_src = spherical_coords[ surface_mask.astype(int), : ]

    # get spherical coordinates from cortical mesh vertex coordinates
    lats_src, lons_src = spherical_coords_src[:,1]-np.pi/2, spherical_coords_src[:,2]
    lats, lons = spherical_coords[:,1]-np.pi/2, spherical_coords[:,2]

    # create mesh data structure
    mesh = stripy.sTriangulation(lons_src, lats_src)

    # interpolate over the sphere
    interp_val, interp_type = mesh.interpolate(lons,lats, zdata=surface_val[surface_mask], order=1)
        
    profiles[:,i] = interp_val
    pd.DataFrame(profiles).to_csv(profile_fn, index=False, header=False)


def transform_surf_to_slab(out_dir,brain,hemi,tfm_list, surf_mid_fn, surf_wm_fn, surf_gm_fn, nslabs, clobber=0):
    surf_mid_list=[]
    surf_wm_list=[]
    surf_gm_list=[]

    for i, slab in enumerate( range(1,1+nslabs) ) :
        surf_mid_rsl_fn=f'{out_dir}/{brain}_{hemi}_{slab}_mid_rsl.obj'
        surf_gm_rsl_fn=f'{out_dir}/{brain}_{hemi}_{slab}_gm_rsl.obj'
        surf_wm_rsl_fn=f'{out_dir}/{brain}_{hemi}_{slab}_wm_rsl.obj'
        
        if not os.path.exists(surf_mid_rsl_fn) or clobber >= 1 or True  : 
            apply_ants_transform_to_obj(surf_mid_fn, tfm_list[i], surf_mid_rsl_fn, [0,0])
            print('Display ', surf_mid_rsl_fn)
        print('exit')
        exit(0)
        if not os.path.exists(surf_wm_rsl_fn) or clobber >= 1  or True: 
            apply_ants_transform_to_obj(surf_mid_fn, tfm_list[i], surf_mid_rsl_fn, [0,0])

        if not os.path.exists(surf_gm_rsl_fn) or clobber >= 1  or True: 
            apply_ants_transform_to_obj(surf_mid_fn, tfm_list[i], surf_mid_rsl_fn, [0,0])
        
        surf_mid_list.append(surf_mid_rsl_fn)
        surf_wm_list.append(surf_wm_rsl_fn)
        surf_gm_list.append(surf_gm_rsl_fn)
    return surf_mid_list, surf_wm_list, surf_gm_list

def surface_interpolation(nl_tfm_str,  slab_str, out_dir, brain, hemi, n_depths, obj_str='civet/mri1/surfaces/mri1_{}_surface_right_{}{}.obj', n_vertices = 327696 clobber=False)
    if not os.path.exists(out_dir) : os.makedirs(out_dir)

    array_src_h5 = f'{out_dir}/{brain}_{hemi}_array_src.h5'
    mask_src_h5  = f'{out_dir}/{brain}_{hemi}_array_mask.h5'
    array_src_nii = f'{out_dir}/{brain}_{hemi}_array_src.nii.gz'
    mask_src_nii  = f'{out_dir}/{brain}_{hemi}_array_mask.nii.gz'
    profile_fn  = f'{out_dir}/{brain}_{hemi}_profiles.csv'
    interp_fn  = f'{out_dir}/{brain}_{hemi}_interp.nii.gz'

    #Interpolate at coordinate locations
    
    #n_vertices = 163848
    #n_vertices = 81920
    surf_mid_fn = obj_str.format('mid', n_vertices,'')
    surf_gm_fn = obj_str.format('gray', n_vertices,'')
    surf_wm_fn = obj_str.format('white', n_vertices,'')
    sphere_obj_fn = obj_str.format('mid', n_vertices,'_sphere')

    #affine = write_mask_and_array(array_src_h5, mask_src_h5, clobber=args.clobber)
    #steps = [ affine[0,0], affine[1,1], affine[2,2]   ]
    #starts = [ affine[0,3], affine[1,3], affine[2,3]   ]
    #array_src = h5.File(array_src_h5, 'r+')
    #mask_src = h5.File(mask_src_h5, 'r+')
    #write_nii( array_src, array_src_nii, affine, np.float32, args.clobber)
    #write_nii( mask_src,  mask_src_nii, affine, np.int32, args.clobber)
    #dimensions = array_src['data'][:].shape
    #mask_src.close()
    #array_src.close()

    print('Get surface mask and surface values')
    surface_mask_fn = f'surface_slab_mask_{n_vertices}.txt'
    surface_val_fn  = f'surface_slab_val_{n_vertices}.txt'
    tfm_list=[]
    slab_list=[]
    nslabs=1 #len(glob(args.nl_tfm_str.format('*','*')))
    lin_df = pd.read_csv(lin_df_fn)
    print('N Slabs:',nslabs)
    for slab in range(1,1+nslabs) :
        lin_fn = lin_df['tfm'].loc[ lin_df['slab'].astype(int) == slab  ].values[0]
        nl_fn = nl_tfm_str.format(slab,slab)
        tfm_list.append([ lin_fn, nl_fn ])
        slab_list.append(slab_str.format(slab,slab))
    surf_mid_list, surf_wm_list, surf_gm_list = transform_surf_to_slab(out_dir,brain,hemi,tfm_list,surf_mid_fn,surf_gm_fn,surf_wm_fn,nslabs)

    if not os.path.exists(profile_fn) or clobber >= 1 :
        get_profiles(surf_mid_list, surf_wm_list, surf_gm_list, n_depths, profile_fn, slab_list )

        profiles = pd.read_csv(profile_fn).values

    mapper = SurfaceVolumeMapper(white_surf=surf_wm_fn, gray_surf=surf_gm_fn, resolution=steps, mask=None,dimensions=dimensions, origin=starts, filename=None, save_in_absence=False )

    print('Map Vector to Block')
    vol_interp = mapper.map_profiles_to_block(profiles)

    print('\tWrite volumetric interpolated values')
    nib.Nifti1Image(vol_interp, affine).to_filename(interp_fn)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--clobber', dest='clobber', type=int, default=0, help='Clobber results')
    parser.add_argument('--brain', dest='brain', type=str, help='brain')
    parser.add_argument('--hemi', dest='hemi', type=str,  help='hemi')
    parser.add_argument('--out-dir', dest='out_dir', type=str,  help='Clobber results')
    parser.add_argument('--nl-tfm-str', dest='nl_tfm_str', type=str,  help='Clobber results')
    parser.add_argument('--lin-df-fn', dest='lin_df_fn', type=str,  help='Clobber results')
    parser.add_argument('--slab-str', dest='slab_str', type=str,  help='Clobber results')
    parser.add_argument('--n-depths', dest='n_depths', type=int,  default=8, help='Clobber results')
    args = parser.parse_args()
    
    lin_df_fn = args.lin_df_fn
    out_dir = args.out_dir
    brain = args.brain
    hemi = args.hemi
    n_depths = args.n_depths
    nl_tfm_str = args.nl_tfm_str
    slab_str = args.slab_str
    clobber=args.clobber
