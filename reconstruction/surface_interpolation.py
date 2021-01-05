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
import ants
import tempfile
import time
import json
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
from utils.utils import shell, w2v, v2w, memory_useage
from utils.upsample_gifti import *
from vast.surface_volume_mapper import SurfaceVolumeMapper
from pykrige.ok import OrdinaryKriging
from sklearn.gaussian_process import GaussianProcessRegressor

global surf_fn_str
surf_fn_str='{}/mri1_{}_surface_right_{}{}.surf.gii'



def krige(lons, lats, lats_src, lons_src, values ):
    # Make this example reproducible:
    # Generate a regular grid with 60° longitude and 30° latitude steps:
    # Create ordinary kriging object:
    memory_useage()
    '''
    OK = OrdinaryKriging(
        lons_src.astype(np.float16),
        lats_src.astype(np.float16),
        values.astype(np.float16),
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic"
        )
    # Execute on grid:
    z1, ss1 = OK.execute("points", lons, lats)
    '''
    X0 = np.array([lons_src,lats_src]).T
    X1 = np.array([lons,lats]).T

    gpr = GaussianProcessRegressor().fit(X0, values.astype(np.float16))
    print("Finished creating regressor")
    d=1000
    z1=np.zeros(X1.shape[0])
    for i in range(0,X1.shape[0],d):
        j=i+d
        z1[i:j] = gpr.predict(X1[i:j]).reshape(-1,)
    print("finished regressing")
     
    return z1
 
def apply_ants_transform_to_gii( in_gii_fn, tfm_list, out_gii_fn, invert):
    print("transforming", in_gii_fn)
    faces, coords = nib.load(in_gii_fn).agg_data(('triangle', 'pointset'))

    tfm = ants.read_transform(tfm_list[0])
    flip = 1
    if np.sum(tfm.fixed_parameters) != 0 : flip=-1
    
    in_file = open(in_gii_fn, 'r')
    
    out_path, out_ext = os.path.splitext(out_gii_fn)
    coord_fn = out_path+ '_ants_reformat.csv'

    #read the csv with transformed vertex points
    with open(coord_fn, 'w+') as f :  
        f.write('x,y,z,t,label\n') 
        for x,y,z in coords :  
            #f.write('{},{},{},{},{}\n'.format(flip*x,flip*y,z,0,0 ))
            f.write('{},{},{},{},{}\n'.format(x,y,z,0,0 ))

    temp_out_fn=tempfile.NamedTemporaryFile().name+'.csv'
    shell(f'antsApplyTransformsToPoints -d 3 -i {coord_fn} -t [{tfm_list[0]},{invert[0]}]  -o {temp_out_fn}',verbose=True)

    # save transformed surfaced as an gii file
    with open(temp_out_fn, 'r') as f :
        #read the csv with transformed vertex points
        for i, l in enumerate( f.readlines() ):
            if i == 0 : continue
            x,y,z,a,b = l.rstrip().split(',')
            coords[i-1] = [flip*float(x),flip*float(y),float(z)]
    
    save_gii(coords,faces,in_gii_fn,out_gii_fn)
    
    obj_fn = out_path+ '.obj'
    print(obj_fn)
    save_obj(obj_fn,coords,faces)

def upsample_and_inflate_surfaces(surf_dir, wm_surf_fn, gm_surf_fn, resolution, depth_list, n_vertices=81920) :
    depth_fn_dict={}
    # Upsampling of meshes at various depths across cortex produces meshes with different n vertices.
    # To create a set of meshes across the surfaces across the cortex that have the same number of 
    # vertices, we first upsample and inflate the wm mesh.
    # Then, for each mesh across the cortex we resample that mesh so that it has the same polygons
    # and hence the same number of vertices as the wm mesh.
    # Each mesh across the cortical depth is inflated (from low resolution, not the upsampled version)
    # and then resampled so that it has the high resolution number of vertices.

    # create depth mesh
    gm_mesh = nib.load(gm_surf_fn) 
    wm_mesh = nib.load(wm_surf_fn)
   
    gm_coords = gm_mesh.agg_data('NIFTI_INTENT_POINTSET')
    gm_faces =  gm_mesh.agg_data('NIFTI_INTENT_TRIANGLE')

    wm_coords = wm_mesh.agg_data('NIFTI_INTENT_POINTSET')
    wm_faces =  wm_mesh.agg_data('NIFTI_INTENT_TRIANGLE')

    d_coords = wm_coords - gm_coords 
    
    gm_upsample_fn="{}/surf_{}mm_{}_rsl.surf.gii".format(surf_dir,resolution,0)
    gm_upsample_csv="{}/surf_{}mm_{}_rsl.surf.csv".format(surf_dir,resolution,0)
    gm_sphere_fn = "{}/surf_{}mm_{}_inflate.surf.gii".format(surf_dir,resolution,0)
    gm_sphere_rsl_fn = "{}/surf_{}mm_{}_inflate_rsl.surf.gii".format(surf_dir,resolution,0)

    depth_fn_dict[0]={'upsample_fn':gm_upsample_fn, 'sphere_rsl_fn':gm_sphere_rsl_fn}

    if False in [ os.path.exists(fn) for fn in [gm_upsample_fn, gm_sphere_fn, gm_sphere_rsl_fn]] :
        create_high_res_sphere(gm_surf_fn, gm_upsample_fn, gm_sphere_fn, gm_sphere_rsl_fn, resolution, gm_upsample_csv)

    for depth in depth_list :
        print("\tDepth", depth)
        depth_surf_fn="{}/surf_{}mm_{}.surf.gii".format(surf_dir,resolution,depth)
        upsample_fn="{}/surf_{}mm_{}_rsl.surf.gii".format(surf_dir,resolution,depth)
        sphere_fn = "{}/surf_{}mm_{}_inflate.surf.gii".format(surf_dir,resolution,depth)
        sphere_rsl_fn = "{}/surf_{}mm_{}_inflate_rsl.surf.gii".format(surf_dir,resolution,depth)
        coords = gm_coords + depth * d_coords
        depth_fn_dict[depth]={'upsample_fn':upsample_fn, 'sphere_rsl_fn':sphere_rsl_fn}

        if not os.path.exists(depth_surf_fn) :
            save_gii( coords, gm_faces, gm_surf_fn, depth_surf_fn)
   
        if False in [ os.path.exists(fn) for fn in [upsample_fn, sphere_fn, sphere_rsl_fn]] :
            create_high_res_sphere(depth_surf_fn, upsample_fn, sphere_fn, sphere_rsl_fn, resolution, optional_reference=gm_upsample_csv)


    return depth_fn_dict

def load_slabs(slab_fn_list) :
    slabs=[]
    for slab_fn in slab_fn_list :
        slabs.append( nib.load(slab_fn) )
    return slabs

def get_valid_idx(src,coords,affine):
    steps = [ affine[0,0], affine[1,1], affine[2,2]   ]
    starts =  [ affine[0,3], affine[1,3], affine[2,3]   ] 
    xmax=starts[0] + src.shape[0] * steps[0]
    ymax=starts[1] + src.shape[1] * steps[1]
    zmax=starts[2] + src.shape[2] * steps[2]
    vx = np.array([ w2v(i, steps[0], starts[0]) for i in coords[:,0]]).astype(int)
    vy = np.array([ w2v(i, steps[1], starts[1]) for i in coords[:,1]]).astype(int)
    vz = np.array([ w2v(i, steps[2], starts[2]) for i in coords[:,2]]).astype(int)
    idx = (vx >= 0) & (vy >= 0) & (vz >= 0) & (vx < src.shape[0]) & ( vy < src.shape[1]) & ( vz < src.shape[2] )
    return idx, (vx,vy,vz)

def vol_surf_interp(src, coords, affine,  clobber=0 ):
    idx , (vx,vy,vz)= get_valid_idx(src,coords,affine)
    
    assert np.sum(idx) != 0 , 'Error: no voxels found to interpolate over'

    val=np.zeros(vx.shape)
    val[idx] = src[vx[idx], vy[idx], vz[idx]]#.reshape(-1,)
    assert np.sum(val) != 0, "Error: no surface values projected onto surface"
    return val


def write_nii(ar, fn, affine, dtype, clobber):
    if not os.path.exists(fn) or clobber >= 1 :
        nib.Nifti1Image(ar['data'][:].astype(dtype), affine).to_filename(fn)

def thicken_sections(array_src, slab_df, resolution ):
    width = np.round(resolution/(0.02*2)).astype(int)
    print('\t\tThickening sections to ',0.02*width*2+0.02)
    dim=[array_src.shape[0], 1, array_src.shape[2]]
    rec_vol = np.zeros_like(array_src)
    for row_i, row in slab_df.iterrows() : 
        i = int(row['volume_order'])
        
        # Conversion of radioactivity values to receptor density values
        section = array_src[:, i, :].reshape(dim) * row['conversion_factor']

        assert np.sum(section) !=0, f'Error: empty frame {i}'
        i0 = (i-width) if i-width > 0 else 0
        i1 = (i+width) if i+(width+1) <= array_src.shape[1] else array_src.shape[1]
        
        #put ligand sections into rec_vol
        rec_vol[:, i0:i1, :] = np.repeat(section,i1-i0, axis=1)
        
    assert np.sum(rec_vol) != 0, 'Error: receptor volume for single ligand is empty'

    return rec_vol


def get_slab_profile( slab_df,  surf_upsample_fn, array_src, affine, profile, resolution):
    rec_vol = thicken_sections(array_src, slab_df, resolution)
    coords = nib.load(surf_upsample_fn).agg_data('NIFTI_INTENT_POINTSET')
    profile += vol_surf_interp( rec_vol, coords, affine, clobber=2)
    assert np.sum(profile) != 0, "Error: empty profile"
    
    return profile


def volume_to_surface_interpolate(depth_fn_list, depth_fn_slab_space, depth_list, slab_dict, df_ligand, resolution, nrows ):
    if False in [os.path.exists(fn) for fn in depth_fn_list ] :
        profiles=np.zeros([nrows, len(depth_list)])
        for i, slab in slab_dict.items() :
            print("loading volume", slab['nl_2d_vol_fn']) 
            array_img = nib.load(slab['nl_2d_vol_fn'])
            array_src = array_img.get_fdata()

            assert np.sum(array_src) != 0 , 'Error: input receptor volume has is empty {}'.format(slab['nl_2d_vol_fn'])

            for depth_index, (depth, depth_fn) in enumerate(zip(depth_list,depth_fn_list)):
                surf_upsample_fn = depth_fn_slab_space[i][depth]

                print(f'\tslab: {i}')
                print(surf_upsample_fn)
                slab_df=df_ligand.loc[df_ligand['slab'].astype(int)==int(i)]
            
                profiles[:,depth_index] = get_slab_profile( slab_df, surf_upsample_fn, array_src, array_img.affine, profiles[:,depth_index], resolution)

        for depth_index, depth_fn in enumerate(depth_fn_list):
            pd.DataFrame(profiles[:,depth_index]).to_csv(depth_fn, index=False, header=False)

def concat_profiles_over_slabs(profiles_fn, ligand, depth,  n_vertices):
    profiles = h5.File(profiles_fn, 'r')
    n = np.zeros(n_vertices)
    concat_profile = np.zeros(n_vertices)
    print('\tConcatenating slabs',profiles[ligand][str(depth)].keys())
    for slab in profiles[ligand][str(depth)].keys() :
        profile = profiles[ligand][str(depth)][slab][:]
        mask = np.zeros(profile.shape[0])
        mask[ profile > 0 ] = 1
        n += mask
        concat_profile += profile
    concat_profile = concat_profile / n
    profiles.close()
    return concat_profile      

def get_profiles(surf_dir, depth_list, depth_fn_slab_space, profiles_fn, profiles_h5_fn, slab_dict, df_ligand, depth_fn_space_mni,  coords_index_dict, resolution, ligand, clobber=False):
    memory_useage()
    example_depth_fn=depth_fn_space_mni[depth_list[0]]['upsample_fn']
    nrows = nib.load(example_depth_fn).agg_data('NIFTI_INTENT_POINTSET').shape[0]

    depth_fn_list = [ sub('.csv', f'_{depth}_raw.csv', profiles_fn) for depth in depth_list ]

    #Project volume onto cortical surface mesh
    volume_to_surface_interpolate(depth_fn_list,depth_fn_slab_space, depth_list, slab_dict, df_ligand, resolution, nrows)

    concatenated_profiles = np.zeros([nrows, len(depth_list)])
    for depth_index, (depth, depth_fn) in enumerate(zip(depth_list, depth_fn_list)):
        print('Depth\t',depth)
        profiles_raw = pd.read_csv(depth_fn,header=None,index_col=None).values
        assert np.sum(profiles_raw) != 0 , 'Error: raw profiles sum to 0 in {}'.format( depth_fn )
        sphere_rsl_fn = depth_fn_space_mni[depth]['sphere_rsl_fn'] 

        print('Depth fn:',depth_fn)
        interpolate_over_surface(sphere_rsl_fn, profiles_h5_fn, depth, profiles_raw, df_ligand, coords_index_dict)
        concatenated_profiles[:,depth_index] = concat_profiles_over_slabs(profiles_h5_fn, ligand, depth, nrows)

    pd.DataFrame(concatenated_profiles).to_csv(profiles_fn, index=False, header=False)
     
def interpolate_over_surface(sphere_obj_fn, profiles_h5_fn, depth, surface_val,df,coords_index_dict):
    memory_useage()
    # get coordinates from dicitonary with mesh info
    profiles = h5.File(profiles_h5_fn, 'r+')
    coords = nib.load(sphere_obj_fn).agg_data('NIFTI_INTENT_POINTSET') #surface_tools.spherical_np(sphere['coords'])
    spherical_coords = surface_tools.spherical_np(coords)
    del coords

    # create mesh data structure
    lats, lons = spherical_coords[:,1]-np.pi/2, spherical_coords[:,2]
    del spherical_coords

    # interpolate over the sphere
    interp_val = np.zeros(lats.shape[0])

    #Interpolate within slabs
    for (ligand, slab, section), df_ligand in df.groupby(['ligand','slab','global_order']):
        # Get maximum section number for a ligand and slab
        section_max =  df['global_order'].loc[ (df['ligand']==ligand) & (df['slab']==slab) ].max()
        print('section max', section_max)

        # Can't interpolate past max section for slab
        if section == section_max : continue
        
        #idx is a boolean list where True means that a vertex is within two autoradiographs
        surface_slab_mask = np.array(coords_index_dict[ligand][str(int(section))]).astype(int)

        # Within valid vertices, define a mask of verticies where we have receptor densities
        #idx = surface_val[surface_slab_mask] != 0 
        idx = surface_val[surface_slab_mask] != 0
        idx = idx.reshape(-1,)
        surface_src_mask = surface_slab_mask[ idx ]
        surface_src_mask = surface_src_mask.reshape(-1,)
        
        # 
        #surface_src_mask = surface_slab_mask[ idx ]
        assert np.sum(surface_src_mask) != 1, "Error, empty profiles {}".format(np.sum(surface_src_mask))

        # get coordinates for vertices in mask
        lats_slab, lons_slab = lats[ surface_slab_mask ], lons[ surface_slab_mask ]
        lats_src = lats[ surface_src_mask]
        lons_src = lons[ surface_src_mask]
        
        # get spherical coordinates from cortical mesh vertex coordinates
        # get values from profiles h5
        interp_values = profiles[ligand][str(depth)][str(slab)][surface_slab_mask]
        n_values = interp_values.shape[0]
        print('ligand',ligand,'\tdepth', depth, '\tslab',slab, '\tsection',section, '\tn', n_values, end='\t')
        if np.sum( surface_val[surface_src_mask] ) == 0:
            if np.sum(interp_values) == 0 : 
                print("Error: No values to interpolate over.")
                exit(1)
            else :
                print('Kriging!')
                interp_values = krige(lons_slab, lats_slab, lats_src, lats_src, surface_val[surface_src_mask])
                profiles[ligand][str(depth)][str(slab)][surface_slab_mask] = interp_values
        else : print('Already interpolated.')

        del interp_values
        del lats_slab
        del lons_slab
        del lats_src
        del lons_src
        del idx
        del surface_src_mask
    
    #Interpolate between slabs


def read_gifti(fn):
    return nib.load( fn ).agg_data('NIFTI_INTENT_POINTSET')

def transform_surf_to_slab(interp_dir, slab_dict, depth_fn_space_mni, df, resolution, clobber=0):
    return_early = True
    surf_rsl_dict={}
    nrows = read_gifti(depth_fn_space_mni[0]['upsample_fn'] ).shape[0]
    slab_val=np.zeros(nrows) 
   
    for slab, cur_slab_dict in slab_dict.items() :
        surf_rsl_dict[slab]={}
        for depth, depth_dict in depth_fn_space_mni.items() :
            upsample_fn = depth_dict['upsample_fn']
            upsample_slab_space_fn="{}/slab-{}_{}".format(interp_dir,slab,os.path.basename(upsample_fn))
            surf_rsl_dict[slab][depth]=upsample_slab_space_fn
            if not os.path.exists(upsample_slab_space_fn) :
                return_early=False

    #hdf5 file to save indices for each section
    section_index_fn=f"{interp_dir}/section_indices.h5"
    if os.path.exists(section_index_fn) :
        section_index = h5.File(section_index_fn, 'r') 
    else : 
        return_early = False

    if return_early : return surf_rsl_dict, section_index
   
    if os.path.exists(section_index_fn) :
        os.remove(section_index_fn)
    section_index = h5.File(section_index_fn, 'w')

    for ligand, df_ligand in df.groupby(['ligand']):
        ligand_f = section_index.create_group(ligand)
        for global_order, df_order in df_ligand.groupby(['global_order']):
            ligand_f.create_dataset(str(global_order),(25000,),dtype='i')

    #an array to save the global order number that is used to represent each vertex
    output_index_ar = np.zeros(coords.shape[1]) 
    
    #sort the data frame by global order so that it is guaranteed to be in correct order
    df.sort_values(['global_order'], inplace=True)

    for slab, cur_slab_dict in slab_dict.items() :

        for depth, depth_dict in depth_fn_space_mni.items() :
            upsample_fn = depth_dict['upsample_fn']
            print(f"\t{slab} {depth}")
            sphere_rsl_fn = depth_dict['sphere_rsl_fn']
            
            if not os.path.exists(surf_rsl_dict[slab][depth]) or clobber >= 1 : 
                apply_ants_transform_to_gii(upsample_fn, [cur_slab_dict['nl_3d_tfm_fn']], slab_rsl_dict[slab][depth], [0])

        print("Updating slab values", slab)
        surf_upsample_fn = surf_rsl_dict[slab][0]
        coords =  read_gifti(surf_upsample_fn)
        array_img = nib.load( cur_slab_dict['nl_2d_vol_fn'] )
        #Affine coordinates for current section
        section_affine = array_img.affine
        coords_idx=np.arange(coords.shape[0]).astype(int)
        
        #Get data frame for current slab
        df_slab = df.loc[ df['slab'] == int(slab) ]
        #Iterate over the sections
        for ligand, df_slab_ligand in df_slab.groupby(['ligand']):
            print(ligand)
            for i in range(df_slab_ligand.shape[0]-1) :
                j=i+1
                global_order_0 = df_slab_ligand['global_order'].iloc[i].astype(int)
                slab_order_0 =  df_slab_ligand['slab_order'].iloc[i]
                slab_order_1 =  df_slab_ligand['slab_order'].iloc[j]
                # Calculate start and end coordinates 
                ystart = section_affine[1,3] + section_affine[1,1] * slab_order_0
                yend = section_affine[1,3] + section_affine[1,1] * slab_order_1
                # Calculate which coordinates are inside the range between the current and next section 
                coords_in_section = coords_idx[ (ystart <= coords[:,1]) & (coords[:,1] <= yend ) ]
                assert np.sum(coords_in_section) != 0, 'Error: No coordinates in section, sum = {}'.format(coords_in_section)
                print('\t',global_order_0, np.sum(coords_in_section != 0) )
                # Save coordinate indices
                section_index[ligand][str(global_order_0)]=coords_in_section.astype(int)

                output_index_ar[(ystart <= coords[:,1]) & (coords[:,1] <= yend )] = global_order_0 

    with open(f"{interp_dir}/vertex_section_indices.txt") as f :
        for i in output_index_ar : f.write('{}\n'.format(output_index_ar[i]))

    return surf_rsl_dict, section_index

def setup_profiles_h5(interp_dir, df, depth_list, n_vertices):
    #Create hdf5 file to store profiles within ligand --> depth --> slab
    profiles_fn = interp_dir+os.sep+'profiles.h5'
    if not os.path.exists(profiles_fn) :
        profiles = h5.File(profiles_fn, 'w')
        for ligand, df_ligand in df.groupby(['ligand']):
            profiles.create_group(ligand)
            for depth in depth_list :
                profiles[ligand].create_group(str(depth))
                for slab, temp in df_ligand.groupby(['slab']):
                    profiles[ligand][str(depth)].create_dataset(str(slab),(n_vertices,),dtype='f')
        profiles.close()

    return profiles_fn

def surface_interpolation(slab_dict, out_dir, interp_dir, brain, hemi, resolution, df, mni_fn, n_depths=3, surf_dir='civet/mri1/surfaces/surfaces/', n_vertices = 327696, clobber=0):
    memory_useage()
    #make sure resolution is interpreted as float
    resolution=float(resolution) 

    surf_rsl_dir = interp_dir +'/surfaces/' 
    os.makedirs(interp_dir, exist_ok=True)
    os.makedirs(surf_rsl_dir, exist_ok=True)
    
    #Interpolate at coordinate locations
    surf_gm_fn = surf_fn_str.format(surf_dir,'gray', n_vertices,'')
    surf_wm_fn = surf_fn_str.format(surf_dir,'white', n_vertices,'')
    sphere_obj_fn = surf_fn_str.format(surf_dir,'mid', n_vertices,'_sphere')

    print("\tGet surface mask and surface values")
    # Load dimensions for output volume
    starts = np.array([-72, -126,-90 ])
    mni_vol = nib.load(mni_fn).get_fdata()
    dwn_res = 0.25
    nat_res = 0.02
    #set depths
    dt = 1.0/ n_depths
    depth_list = np.arange(dt, 1+dt, dt)

    dimensions = np.array([ mni_vol.shape[0] * dwn_res/resolution, 
                            mni_vol.shape[1] * dwn_res/resolution, 
                            mni_vol.shape[2] * dwn_res/resolution]).astype(int)
    del mni_vol

    print("\tUpsample and Inflate Surfaces") 

    #upsample transformed surfaces to given resolution
    depth_fn_space_mni = upsample_and_inflate_surfaces(surf_rsl_dir, surf_wm_fn, surf_gm_fn, resolution, depth_list)
    
    #For each slab, transform the mesh surface to the receptor space
    #TODO: transform points into space of individual autoradiographs
    df_brain_hemi = df.loc[ (df['mri'] == brain) & (df['hemisphere'] == hemi) ]
    memory_useage()
    print('\tTransforming surface to slab...',end=' ')
    depth_fn_slab_space, coords_index_dict = transform_surf_to_slab(surf_rsl_dir, slab_dict, depth_fn_space_mni, df_brain_hemi, resolution)
    print('Done.')
    memory_useage()
    
    # Create an object that will be used to interpolate over the surfaces
    mapper = SurfaceVolumeMapper(white_surf=surf_wm_fn, gray_surf=surf_gm_fn, resolution=[resolution]*3, mask=None, dimensions=dimensions, origin=starts, filename=None, save_in_absence=False, out_dir=interp_dir, left_oriented=True )
    
    depth_list = np.insert(depth_list,0, 0)

    #setup h5 file in which to store surface profiles for each slab
    n_vertices = read_gifti(depth_fn_space_mni[0]['upsample_fn']).shape[0]
    profiles_h5_fn = setup_profiles_h5(interp_dir, df, depth_list, n_vertices)

    memory_useage()
    #Iterate over ligands, interpolating missing sections for each type one at a time
    for ligand, df_ligand in df_brain_hemi.groupby(['ligand']):
        print('\tInterpolating for ligand:',ligand)
        profiles_fn  = f'{interp_dir}/{brain}_{hemi}_{ligand}_{resolution}mm_profiles.csv'
        interp_fn  = f'{interp_dir}/{brain}_{hemi}_{ligand}_{resolution}mm.nii.gz'

        # Extract profiles from the slabs using the surfaces 
        if not os.path.exists(profiles_fn) or clobber >= 1 :
            print('Getting profiles')
            get_profiles( surf_rsl_dir, depth_list, depth_fn_slab_space,  profiles_fn,  profiles_h5_fn, slab_dict, df_ligand, depth_fn_space_mni,  coords_index_dict, resolution, ligand )
            
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
            print('Done')


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
