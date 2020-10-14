import os
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import pandas as pd
import stripy as stripy
import numpy as np
import vast.surface_tools as surface_tools
import vast.surface_tools as io_mesh
from nibabel.processing import resample_to_output
from utils.apply_ants_transform_to_obj import apply_ants_transform_to_obj
from re import sub
from glob import glob
from utils.mesh_io import load_mesh_geometry, save_mesh_data, save_obj, read_obj
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_fill_holes
from nibabel.processing import resample_to_output
from skimage.filters import threshold_otsu, threshold_li
from ants import  from_numpy,  apply_transforms, apply_ants_transform, read_transform
from ants import image_read, registration
from utils.utils import shell, w2v, v2w
from vast.surface_volume_mapper import SurfaceVolumeMapper

def load_slabs(slab_fn_list) :
    slabs=[]
    for slab_fn in slab_fn_list :
        slabs.append( nib.load(slab_fn) )
    return slabs



def vol_surf_interp(src, coords, affine,  clobber=0 ):

    steps = [ affine[0,0], affine[1,1], affine[2,2]   ]
    starts = [-72, -126,-90 ] # [ affine[0,3], affine[1,3], affine[2,3]   ]

    vx=np.array([ w2v(i, steps[0], starts[0]) for i in coords[:,0]]).astype(int)
    vy=np.array([ w2v(i, steps[1], starts[1]) for i in coords[:,1]]).astype(int)
    vz=np.array([ w2v(i, steps[2], starts[2]) for i in coords[:,2]]).astype(int)
    
    idx = (vx >= 0) & (vy >= 0) & (vz >= 0) & (vx < src.shape[0]) & ( vy < src.shape[1]) & ( vz < src.shape[2] )
    if np.sum(idx) == 0 : 
        print('Error: no voxels found to interpolate over')
        exit(1)

    val=np.zeros(vx.shape)
    val[idx] = src[vx[idx], vy[idx], vz[idx]].reshape(-1,)
    df = pd.DataFrame({'x':coords[:,0], 'y':coords[:,1],'z':coords[:,2],'vx':vx,'vy':vy,'vz':vz,'val':val})
    return val


def write_nii(ar, fn, affine, dtype, clobber):
    if not os.path.exists(fn) or clobber >= 1 :
        nib.Nifti1Image(ar['data'][:].astype(dtype), affine).to_filename(fn)


def get_profiles(sphere_obj_fn, surf_mid_list, surf_wm_list, surf_gm_list, n_depths,  profile_fn, slab_list, df_ligand ):
    dt = 1.0/ n_depths
    depth_list = np.arange(0., 1+dt, dt)
    
    nrows = pd.read_csv(surf_mid_list[0]).shape[0]

    profiles=np.zeros([nrows, len(depth_list)])

    surface_mask=np.zeros(nrows)
    surface_val=np.zeros(nrows)

    nslabs=len(slab_list)
    for slab, slab_df in df_ligand.groupby(['slab']) :

        i = int(slab) - 1

        surf_mid_dict = pd.read_csv(surf_mid_list[i])
        surf_gm_dict =  pd.read_csv(surf_gm_list[i])
        surf_wm_dict =  pd.read_csv(surf_wm_list[i])
        
        gm_coords = surf_gm_dict.loc[ :, ['x','y','z'] ].values
        wm_coords = surf_wm_dict.loc[ :, ['x','y','z'] ].values
        
        d_coords = gm_coords - wm_coords
        print('\tobj',surf_mid_list[i])
        print('\tnii',slab_list[i])
        array_img = nib.load(slab_list[i])
        array_src = array_img.get_fdata()
        rec_vol = np.zeros_like(array_src)

        #put ligand sections into rec_vol
        rec_vol[:, slab_df['volume_order'].values.astype(int), :] = array_src[:, slab_df['volume_order'].values.astype(int), :]
        
        for i, depth in enumerate(depth_list) :
            print('\tDepth',depth)
            #
            coords = wm_coords + depth * d_coords
            #
            temp_surface_val = vol_surf_interp( rec_vol, coords, array_img.affine, clobber=1)
            print(temp_surface_val.shape, surface_val.shape)
            surface_val += temp_surface_val
        
    #define a mask of verticies where we have receptor densitiies
    surface_mask = surface_val != 0 
    #define vector with receptor densities 
    surface_val_src = surface_val[ surface_mask.astype(bool) ]
    #define vector without receptor densities
    surface_val_tgt = surface_val[ ~surface_mask.astype(bool) ]

    print('Convert to sphereical coordinates')
    #load sphere mesh
    sphere = load_mesh_geometry(sphere_obj_fn) 
    # get coordinates from dicitonary with mesh info
    spherical_coords = surface_tools.spherical_np(sphere['coords'])
    # get coordinates for vertices in mask
    spherical_coords_src = spherical_coords[ surface_mask.astype(bool), : ]

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
        surf_mid_rsl_csv=f'{out_dir}/{brain}_{hemi}_{slab}_mid_rsl.csv'
        surf_gm_rsl_csv=f'{out_dir}/{brain}_{hemi}_{slab}_gm_rsl.csv'
        surf_wm_rsl_csv=f'{out_dir}/{brain}_{hemi}_{slab}_wm_rsl.csv'
        print('Applying transform', tfm_list[i])
        if not os.path.exists(surf_mid_rsl_fn) or clobber >= 1 : 
            apply_ants_transform_to_obj(surf_mid_fn, [tfm_list[i]], surf_mid_rsl_fn, [0])
        
        if not os.path.exists(surf_wm_rsl_fn) or clobber >= 1  : 
            apply_ants_transform_to_obj(surf_wm_fn, [tfm_list[i]], surf_wm_rsl_fn, [0])

        if not os.path.exists(surf_gm_rsl_fn) or clobber >= 1  : 
            apply_ants_transform_to_obj(surf_gm_fn, [tfm_list[i]], surf_gm_rsl_fn, [0])
        
        surf_mid_list.append(surf_mid_rsl_csv)
        surf_wm_list.append(surf_wm_rsl_csv)
        surf_gm_list.append(surf_gm_rsl_csv)
    return surf_mid_list, surf_wm_list, surf_gm_list

def surface_interpolation(tfm_list,  slab_list, out_dir, brain, hemi, resolution, df, mni_fn, n_depths=3, surf_dir='civet/mri1/surfaces/surfaces/', n_vertices = 327696, clobber=0):
    #make sure resolution is interpreted as float
    resolution=float(resolution) 

    obj_str='{}/mri1_{}_surface_right_{}{}.obj'
    
    if not os.path.exists(out_dir) : os.makedirs(out_dir)

    #Interpolate at coordinate locations
    surf_mid_fn = obj_str.format(surf_dir,'mid', n_vertices,'')
    surf_gm_fn = obj_str.format(surf_dir,'gray', n_vertices,'')
    surf_wm_fn = obj_str.format(surf_dir,'white', n_vertices,'')
    sphere_obj_fn = obj_str.format(surf_dir,'mid', n_vertices,'_sphere')
    
    print('Sphere obj:', sphere_obj_fn)

    print('Get surface mask and surface values')
    surface_mask_fn = f'surface_slab_mask_{n_vertices}.txt'
    surface_val_fn  = f'surface_slab_val_{n_vertices}.txt'
    nslabs=len(tfm_list) 
    print('N Slabs:',nslabs)
    
    # Load dimensions for output volume
    starts = np.array([-72, -126,-90 ])
    mni_vol = nib.load(mni_fn).get_fdata()
    dimensions = np.array( np.array(mni_vol.shape)  / resolution ).astype(int) 
    del mni_vol

    #For each slab, transform the mesh surface to the receptor space
    #TODO: transform points into space of individual autoradiographs
    surf_mid_list, surf_wm_list, surf_gm_list = transform_surf_to_slab(out_dir, brain, hemi, tfm_list, surf_mid_fn, surf_gm_fn, surf_wm_fn, nslabs)
    
    # Create an object that will be used to interpolate over the surfaces
    mapper = SurfaceVolumeMapper(white_surf=surf_wm_fn, gray_surf=surf_gm_fn, resolution=[resolution]*3, mask=None,dimensions=dimensions, origin=starts, filename=None, save_in_absence=False, out_dir=out_dir )

    for ligand, df_ligand in df.groupby(['ligand']):
        print('\tInterpolating for ligand:',ligand)
        # Extract profiles from the slabs using the surfaces 
        profile_fn  = f'{out_dir}/{brain}_{hemi}_{ligand}_{resolution}mm_profiles.csv'
        if not os.path.exists(profile_fn) or clobber >= 1 :
            get_profiles(sphere_obj_fn, surf_mid_list, surf_wm_list, surf_gm_list, n_depths, profile_fn, slab_list, df_ligand)
        profiles = pd.read_csv(profile_fn).values
            
        # Interpolate a 3D receptor volume from the surface mesh profiles
        interp_fn  = f'{out_dir}/{brain}_{hemi}_{ligand}_{resolution}mm.nii.gz'
        if not os.path.exists(interp_fn) or clobber : 
            print('Map Vector to Block')
            vol_interp = mapper.map_profiles_to_block(profiles)

            print('\tWrite volumetric interpolated values')
            receptor_img = nib.Nifti1Image(vol_interp, np.array([[resolution, 0, 0, starts[0]],
                                                                 [0, resolution, 0, starts[1]],
                                                                 [0, 0, resolution, starts[2]],
                                                                 [0,0,0,1]])
                                                                 )
            receptor_img_rsl = resample_to_output(receptor_img, [resolution]*3)
            receptor_img_rsl.to_filename(interp_fn)



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
