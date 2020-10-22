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

global obj_str
obj_str='{}/mri1_{}_surface_right_{}{}.obj'

def load_slabs(slab_fn_list) :
    slabs=[]
    for slab_fn in slab_fn_list :
        slabs.append( nib.load(slab_fn) )
    return slabs



def vol_surf_interp(src, coords, affine,  clobber=0 ):
    steps = [ affine[0,0], affine[1,1], affine[2,2]   ]
    starts =  [ affine[0,3], affine[1,3], affine[2,3]   ] #[-72, -126,-90 ] #
    vx=np.array([ w2v(i, steps[0], starts[0]) for i in coords[:,0]]).astype(int)
    vy=np.array([ w2v(i, steps[1], starts[1]) for i in coords[:,1]]).astype(int)
    vz=np.array([ w2v(i, steps[2], starts[2]) for i in coords[:,2]]).astype(int)

    idx = (vx >= 0) & (vy >= 0) & (vz >= 0) & (vx < src.shape[0]) & ( vy < src.shape[1]) & ( vz < src.shape[2] )
    if np.sum(idx) == 0 : 
        print('Error: no voxels found to interpolate over')
        exit(1)
    val=np.zeros(vx.shape)
    val[idx] = src[vx[idx], vy[idx], vz[idx]]#.reshape(-1,)
    df = pd.DataFrame({'x':coords[:,0], 'y':coords[:,1],'z':coords[:,2],'vx':vx,'vy':vy,'vz':vz,'val':val})
    return val


def write_nii(ar, fn, affine, dtype, clobber):
    if not os.path.exists(fn) or clobber >= 1 :
        nib.Nifti1Image(ar['data'][:].astype(dtype), affine).to_filename(fn)

def thicken_sections(array_src, slab_df, resolution ):
    width = resolution/(0.02*2)
    print('\t\tThickening sections to ',0.02*width*2)
    dim=[array_src.shape[0], 1, array_src.shape[2]]
    rec_vol = np.zeros_like(array_src)
    for row_i, row in slab_df.iterrows() : 
        i = row['volume_order'].values.astype(int)
        
        # Conversion of radioactivity values to receptor density values
        section = array_src[:, i, :].reshape(dim) * row['conversion_factor'].values[0]

        assert np.sum(section) !=0, f'Error: empty frame {i}'
        i0 = (i-width) if i-width > 0 else 0
        i1 = (i+width) if i+width <= array_src.shape[1] else array_src.shape[1]
        
        #put ligand sections into rec_vol
        rec_vol[:, i0:i1, :] = np.repeat(section,i1-i0, axis=1)
        
    assert np.sum(rec_vol) != 0, 'Error: receptor volume for single ligand is empty'

    return rec_vol

def get_slab_profile(wm_coords,gm_coords, d_coords, slab_df, depth_list, array_src, affine, profiles, resolution):
    
    rec_vol = thicken_sections(array_src, slab_df, resolution)
    nib.Nifti1Image(rec_vol, affine).to_filename('/project/def-aevans/tfunck/test.nii.gz')
    print('\t\t {}\n\t\t Depth:'.format( np.mean(np.sum(np.abs(wm_coords-gm_coords),axis=1)) ) )
    for depth_i, depth in enumerate(depth_list) :
        #print(f' {depth} ',end=' ')
        coords = wm_coords + depth * d_coords
        profiles[:,depth_i] += vol_surf_interp( rec_vol, coords, affine, clobber=1)
        if depth_i == 0 :
            idx = profiles[:,depth_i]>0
        #print('coords',coords[idx][0], profiles[idx,depth_i][0])
    return profiles
                 
def get_profiles(sphere_obj_fn, surf_mid_list, surf_wm_list, surf_gm_list, n_depths,  profiles_fn,vol_list, slab_list, df_ligand, resolution ):
    dt = 1.0/ n_depths
    depth_list = np.arange(0., 1+dt, dt)
    
    nrows = pd.read_csv(surf_mid_list[0]).shape[0]

    profiles=np.zeros([nrows, len(depth_list)])

    for i, slab in enumerate(slab_list) :
        print(f'\tslab: {slab}')
        slab_df=df_ligand.loc[df_ligand['slab'].astype(int)==int(slab)]

        surf_mid_dict = pd.read_csv(surf_mid_list[i])
        surf_gm_dict =  pd.read_csv(surf_gm_list[i])
        surf_wm_dict =  pd.read_csv(surf_wm_list[i])
        gm_coords = surf_gm_dict.loc[ :, ['x','y','z'] ].values
        wm_coords = surf_wm_dict.loc[ :, ['x','y','z'] ].values
        
        d_coords = gm_coords - wm_coords
        
        array_img = nib.load(vol_list[i])
        array_src = array_img.get_fdata()
        assert np.sum(array_src) != 0 , f'Error: input receptor volume has is empym {vol_list[i]}'

        profiles += get_slab_profile(wm_coords, gm_coords, d_coords, slab_df, depth_list, array_src, array_img.affine, profiles, resolution)

    profiles[ profiles < 0.1 ] = 0
   
    profiles_raw_fn = sub('.csv','_raw.csv', profiles_fn) 
    pd.DataFrame(profiles).to_csv(profiles_raw_fn, index=False, header=False)

    profiles = interpolate_over_surface(sphere_obj_fn,profiles)

    pd.DataFrame(profiles).to_csv(profiles_fn, index=False, header=False)
     
def interpolate_over_surface(sphere_obj_fn,profiles):
        
    sphere = load_mesh_geometry(sphere_obj_fn) 

    # get coordinates from dicitonary with mesh info
    spherical_coords = surface_tools.spherical_np(sphere['coords'])

    for i in range(profiles.shape[1]):
        surface_val = profiles[:,i]

        #define a mask of verticies where we have receptor densitiies
        surface_mask = profiles[:,i] != 0

        #define vector with receptor densities 
        surface_val_src = surface_val[ surface_mask.astype(bool) ]

        #define vector without receptor densities
        surface_val_tgt = surface_val[ ~surface_mask.astype(bool) ]

        # get coordinates for vertices in mask
        spherical_coords_src = spherical_coords[ surface_mask.astype(bool), : ]

        # get spherical coordinates from cortical mesh vertex coordinates
        lats_src, lons_src = spherical_coords_src[:,1]-np.pi/2, spherical_coords_src[:,2]

        # create mesh data structure
        mesh = stripy.sTriangulation(lons_src, lats_src)
        lats, lons = spherical_coords[:,1]-np.pi/2, spherical_coords[:,2]

        # interpolate over the sphere
        interp_val, interp_type = mesh.interpolate(lons,lats, zdata=surface_val[surface_mask], order=1)
            
        profiles[:,i] = interp_val

    return profiles

def transform_surf_to_slab(out_dir,brain,hemi,tfm_list, surf_mid_fn, surf_wm_fn, surf_gm_fn, slab_list, clobber=0):
    surf_mid_list=[]
    surf_wm_list=[]
    surf_gm_list=[]

    for i, slab in enumerate( slab_list ) :
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

def surface_interpolation(tfm_list, vol_list, slab_list, out_dir, brain, hemi, resolution, df, mni_fn, n_depths=3, surf_dir='civet/mri1/surfaces/surfaces/', n_vertices = 327696, clobber=0):
    #make sure resolution is interpreted as float
    resolution=float(resolution) 

    
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
    
    # Load dimensions for output volume
    starts = np.array([-72, -126,-90 ])
    mni_vol = nib.load(mni_fn).get_fdata()
    dwn_res = 0.25
    nat_res = 0.02
    dimensions = np.array([ mni_vol.shape[0] * dwn_res/resolution, 
                            mni_vol.shape[1] * dwn_res/resolution, 
                            mni_vol.shape[2] * dwn_res/resolution]).astype(int)
    del mni_vol

    #For each slab, transform the mesh surface to the receptor space
    #TODO: transform points into space of individual autoradiographs
    surf_mid_list, surf_wm_list, surf_gm_list = transform_surf_to_slab(out_dir, brain, hemi, tfm_list, surf_mid_fn, surf_gm_fn, surf_wm_fn, slab_list)
    
    # Create an object that will be used to interpolate over the surfaces
    mapper = SurfaceVolumeMapper(white_surf=surf_wm_fn, gray_surf=surf_gm_fn, resolution=[resolution]*3, mask=None,dimensions=dimensions, origin=starts, filename=None, save_in_absence=False, out_dir=out_dir )

    for ligand, df_ligand in df.groupby(['ligand']):
        print('\tInterpolating for ligand:',ligand)
        profiles_fn  = f'{out_dir}/{brain}_{hemi}_{ligand}_{resolution}mm_profiles.csv'
        interp_fn  = f'{out_dir}/{brain}_{hemi}_{ligand}_{resolution}mm.nii.gz'

        # Extract profiles from the slabs using the surfaces 
        if not os.path.exists(profiles_fn) or clobber >= 1 :
            get_profiles(sphere_obj_fn, surf_mid_list, surf_wm_list, surf_gm_list, n_depths, profiles_fn, vol_list, slab_list, df_ligand, resolution)
            
        # Interpolate a 3D receptor volume from the surface mesh profiles
        if not os.path.exists(interp_fn) or clobber : 

            print('Map Vector to Block')
            profiles = pd.read_csv(profiles_fn, header=None).values
            vol_interp = mapper.map_profiles_to_block(profiles)

            assert np.sum(vol_interp) != 0 , 'Error: interpolated volume is empty'

            receptor_img = nib.Nifti1Image(vol_interp, np.array([[resolution, 0, 0, starts[0]],
                                                                 [0, resolution, 0, starts[1]],
                                                                 [0, 0, resolution, starts[2]],
                                                                 [0, 0, 0, 1]]) )

            print(f'\n\tResample interpolated volume to {resolution}')
            receptor_img_rsl = resample_to_output(receptor_img, [resolution]*3, order=1)

            print(f'\tWrite volumetric interpolated values to {interp_fn} ',end='\t')
            receptor_img.to_filename(interp_fn)
            #receptor_img_rsl.to_filename(interp_fn)
            print('Done')
        exit(0)


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
