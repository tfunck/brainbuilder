import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import utils.ants_nibabel as nib
import nibabel as nb_surf
import h5py as h5
import pandas as pd
import stripy as stripy
import numpy as np
import vast.surface_tools as surface_tools
import ants
import tempfile
import time
import re
from nibabel import freesurfer
from utils.mesh_io import load_mesh
from skimage.transform import resize
from utils.utils import get_section_intervals, apply_ants_transform_to_gii
from utils.combat_slab_normalization import combat_slab_normalization
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label
from nibabel.processing import resample_to_output
from re import sub
from glob import glob
from utils.mesh_io import load_mesh_geometry, save_mesh, save_mesh_data, save_obj, read_obj
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_fill_holes
from nibabel.processing import resample_to_output
from skimage.filters import threshold_otsu, threshold_li
from ants import  from_numpy,  apply_transforms, apply_ants_transform, read_transform
from ants import image_read, registration
from utils.utils import shell, w2v, v2w
from upsample_gifti import save_gii, upsample_gifti, identify_target_edges, obj_to_gii
from vast.surface_volume_mapper import SurfaceVolumeMapper
#from pykrige.ok import OrdinaryKriging
global surf_base_str
surf_base_str = '{}/{}_{}_surface_{}_{}{}.{}'

'''
def krig(lons_src, lats_src, lats, lons, values ):
    # Make this example reproducible:
    #np.random.seed(89239413)

    # Generate random data following a uniform spatial distribution
    # of nodes and a uniform distribution of values in the interval
    # [2.0, 5.5]:

    # Generate a regular grid with 60° longitude and 30° latitude steps:
    
    grid_lon = np.linspace(0.0, 360.0, 7)
    grid_lat = np.linspace(-90.0, 90.0, 7)

    # Create ordinary kriging object:
    OK = OrdinaryKriging(
        lon,
        lat,
        values,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic"
        )

    # Execute on grid:
    z1, ss1 = OK.execute("grid", lons_tgt, lats_tgt)
    
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
'''
 
from ants import get_center_of_mass




def upsample_and_inflate_surfaces(surf_dir, wm_surf_fn, gm_surf_fn, ext, resolution, upsample_resolution,  depth_list, slab_dict, clobber=False, n_vertices=81920):
    depth_fn_dict={}
    # Upsampling of meshes at various depths across cortex produces meshes with different n vertices.
    # To create a set of meshes across the surfaces across the cortex that have the same number of 
    # vertices, we first upsample and inflate the wm mesh.
    # Then, for each mesh across the cortex we resample that mesh so that it has the same polygons
    # and hence the same number of vertices as the wm mesh.
    # Each mesh across the cortical depth is inflated (from low resolution, not the upsampled version)
    # and then resampled so that it has the high resolution number of vertices.

    # create depth mesh
    #gm_mesh = nb_surf.load(gm_surf_fn) 
    #wm_mesh = nb_surf.load(wm_surf_fn)
   
    #gm_coords = gm_mesh.agg_data('NIFTI_INTENT_POINTSET')

    #wm_coords = wm_mesh.agg_data('NIFTI_INTENT_POINTSET')
    #wm_faces =  wm_mesh.agg_data('NIFTI_INTENT_TRIANGLE')

    from nibabel.freesurfer.io import read_geometry, write_geometry, write_annot, write_morph_data
    gm_coords, gm_faces, gm_info = load_mesh(gm_surf_fn)
    wm_coords, wm_faces, wm_info = load_mesh(wm_surf_fn)

    if ext.wm == '.white' : volume_info=gm_info
    else : volume_info = gm_surf_fn
    
    gm_obj_fn="{}/surf_{}mm_{}_rsl.obj".format(surf_dir,resolution,0)
    wm_upsample_fn="{}/surf_{}mm_{}_rsl{}".format(surf_dir,resolution,1, ext.wm)
    gm_sphere_fn = "{}/surf_{}mm_{}{}".format(surf_dir,resolution,0, ext.gm_sphere)
    gm_sphere_rsl_fn = "{}/surf_{}mm_{}rsl{}".format(surf_dir,resolution,0, ext.gm_sphere)
    
    d_coords = wm_coords - gm_coords 
    del wm_coords
    
    input_list = []
    output_list = []

    upsample_0_fn = "{}/surf_{}mm_{}_rsl{}".format(surf_dir,resolution,depth_list[0], ext.gm)
    upsample_1_fn = "{}/surf_{}mm_{}_rsl{}".format(surf_dir,resolution,depth_list[-1], ext.gm)
    
    for depth in depth_list :
        print("\tDepth", depth)
        depth_surf_fn = "{}/surf_{}mm_{}{}".format(surf_dir,resolution,depth, ext.gm)
        upsample_fn = "{}/surf_{}mm_{}_rsl.h5".format(surf_dir,resolution,depth)
        upsample_gii_fn = "{}/surf_{}mm_{}_rsl{}".format(surf_dir,resolution,depth, ext.gm)
        sphere_fn = "{}/surf_{}mm_{}_inflate{}".format(surf_dir,resolution,depth, ext.gm_sphere)
        sphere_rsl_fn = "{}/surf_{}mm_{}_inflate_rsl.h5".format(surf_dir,resolution,depth)
        depth_fn_dict[depth]={'upsample_fn':upsample_fn, 'upsample_gii_fn':upsample_gii_fn, 'sphere_rsl_fn':sphere_rsl_fn}
        
        coords = gm_coords + depth * d_coords
        
        if not os.path.exists(depth_surf_fn) :
            save_mesh(depth_surf_fn, coords, gm_faces, volume_info=volume_info)

        del coords
   
        if not os.path.exists(sphere_fn) or  clobber :
            print('\tInflate to sphere')
            shell('~/freesurfer/bin/mris_inflate -n 500  {} {}'.format(depth_surf_fn, sphere_fn))
            #shell('~/freesurfer/bin/mris_inflate -n 10  {} {}'.format(depth_surf_fn, sphere_fn))
        
        input_list += [depth_surf_fn, sphere_fn]
        output_list+= [upsample_fn, sphere_rsl_fn]

    depth_fn_dict[depth_list[0]]['upsample_gii_fn'] = upsample_0_fn
    depth_fn_dict[depth_list[-1]]['upsample_gii_fn'] = upsample_1_fn

    edge_mask = identify_target_edges(slab_dict,  surf_dir, gm_surf_fn, ext=ext.gm)
    faces_fn, coords_fn = upsample_gifti(gm_surf_fn, upsample_0_fn, upsample_1_fn, float(upsample_resolution), input_list=input_list, output_list=output_list, edge_mask=edge_mask, clobber=clobber)
    
    depth_fn_dict[depth_list[0]]['faces_fn'] = faces_fn

    rsl_faces = h5.File(faces_fn,'r')['data'][:]
    rsl_coords = h5.File(coords_fn, 'r')['data'][:]
    if not os.path.exists(gm_obj_fn) :
        save_obj(gm_obj_fn, rsl_coords,rsl_faces)

    return depth_fn_dict



def get_valid_coords( coords, iw):
    lower = min(iw)
    upper = max(iw)
    valid_coords_idx = (coords[:,1] >= lower) & (coords[:,1] <= upper)
    valid_coords_idx = valid_coords_idx.reshape(valid_coords_idx.shape[0])
    valid_coords = coords[valid_coords_idx, :]

    return valid_coords, valid_coords_idx




def mesh_to_volume(coords, vertex_values, dimensions, starts, steps, origin=[0,0,0], interp_vol=None, n_vol=None ):
    '''
    About
        Interpolate mesh values into a volume
    Arguments
        coords
        vertex_values
        dimensions
        starts
        steps
        interp_vol
        n_vol
    Return
        interp_vol
        n_vol
    '''
    if type(vertex_values) != np.ndarray  or type(n_vol) != np.ndarray :
        interp_vol = np.zeros(dimensions)
        n_vol = np.zeros_like(interp_vol)
    
    coords[:,0] += origin[0]
    coords[:,1] += origin[1]
    coords[:,2] += origin[2]
    print('Origin offset being applied:', origin);
    #coords[:,0] -= 0.04
    #coords[:,1] -= -30.56
    #coords[:,2] += 24.94
    
    x = np.rint( (coords[:,0] - starts[0]) / steps[0] ).astype(int)
    y = np.rint( (coords[:,1] - starts[1]) / steps[1] ).astype(int)
    z = np.rint( (coords[:,2] - starts[2]) / steps[2] ).astype(int)



    idx = (x >= 0) & (y >= 0) & (z >= 0) & (x < dimensions[0]) & ( y < dimensions[1]) & ( z < dimensions[2] )
    
    assert np.sum(idx) > 0, 'Assert: no voxels found inside mesh_to_volume'
    x = x[idx]
    y = y[idx]
    z = z[idx]
    vertex_values = vertex_values[idx] 
    for i, (xc, yc, zc) in enumerate(zip(x,y,z)) :
        interp_vol[xc,yc,zc] += vertex_values[i]
        n_vol[xc,yc,zc] += 1

    return interp_vol, n_vol

def multi_mesh_to_volume(profiles, depth_fn_slab_space, depth_list, dimensions, starts, steps, origin=[0,0,0]):
    
    interp_vol = np.zeros(dimensions)
    n_vol = np.zeros_like(interp_vol)

    for ii in range(profiles.shape[1]):
        try :
            surf_fn = depth_fn_slab_space[depth_list[ii]]['upsample_h5']
        except KeyError:
            surf_fn = depth_fn_slab_space[depth_list[ii]]['upsample_fn']

        coords = h5.File(surf_fn,'r')['data'][:]
        profiles_vtr = profiles[:,ii]
        assert np.sum(profiles_vtr), f'Error, empty profiles in multi_mesh_to_volume in column {ii}, depth {depth_list[ii]}'
        print(starts)
        print(steps)
        print(dimensions);
        interp_vol, n_vol = mesh_to_volume(coords, profiles_vtr, dimensions, starts, steps, interp_vol=interp_vol, n_vol=n_vol, origin=origin )

    interp_vol[ n_vol>0 ] = interp_vol[n_vol>0] / n_vol[n_vol>0]
    
    assert np.sum(interp_vol) != 0 , 'Error: interpolated volume is empty'
    return interp_vol

def project_volumes_to_surfaces(surf_fn_list, thickened_dict, interp_csv, interp_dir, origin=np.array([0,0,0]), clobber=False):
    '''
    About : For a given ligand, interpoalte the autoradiograph receptor densities 
            onto each surface depth for each slab. The interpolation onto the surface is currently 
            performed with nearest neighbour interpolation. 

    Inputs :
        surf_fn_list :  list of surface files onto which ligand values are projected
        thickened_dict :   list of slab volumes with ligand binding densities
        interp_csv :    csv filename where projected values are saved
        interp_dir :    directory name for interpolated csv files

    Outputs :
        None
    '''
    print('interp csv', interp_csv)
    if not os.path.exists(interp_csv) or clobber: 
        qc_dir = interp_dir + '/qc/'
        os.makedirs(qc_dir, exist_ok=True)
        
        nvertices = h5.File(surf_fn_list[0]['upsample_h5'],'r')['data'].shape[0]
        #the default value of the vertices is set to -100 to make it easy to distinguish
        #between vertices where a ligand density of 0 is measured versus the default value
        all_values=np.zeros(nvertices) - 100
        print(surf_fn_list) 
        #Iterate over slabs within a given 
        for i, surf_fn in enumerate(surf_fn_list) :
            vol_fn = thickened_dict[str(i+1)]
           
            # read surface coordinate values from file
            coords_h5 = h5.File(surf_fn['upsample_h5'],'r')
            coords = coords_h5['data'][:]
            coords += origin
           
            #These offsets were used for the macaque brain, but in theory they should
            # be accounted for when transforming the coordinates to slab space and shouldn't
            #be needed
            #coords[:,0] -= 0.04
            #coords[:,1] -= 30.56
            #coords[:,2] += 24.94
            #for i in range(3) : coords[:,i] += origin[i]

            # read slab volume
            img = nib.load(vol_fn)
            vol = img.get_fdata()
            affine = nb_surf.load(vol_fn).affine
            assert np.max(vol) != np.min(vol) , f'Error: empty volume {vol_fn}; {np.max(vol) != np.min(vol)} '

            # set variable names for coordinate system
            xstart = affine[0,3]
            ystart = affine[1,3]
            zstart = affine[2,3]
            xstep = affine[0,0]
            ystep = affine[1,1]
            zstep = affine[2,2]

            # get the intervals along the y-axis of the volume where
            # we have voxel intensities that should be interpolated onto the surfaces
            intervals_voxel = get_section_intervals(vol)
            #convert from voxel values to real world coordinates
            assert len(intervals_voxel) > 0 , 'Error: the length of intervals_voxels == 0 ' 
            #for iv in intervals_voxel : 
            for y0, y1 in intervals_voxel : #range(iv[0],iv[1]+1):
                #y1 = y0 + 1
                y0w = y0 * ystep + ystart 
                y1w = y1 * ystep + ystart 
                #the voxel values should be the same along the y-axis within an interval
                #WARNING: this will NOT work if there isn't a gap between thickened autoradiograph sections!
                section = vol[:,y0,:]
                #assert np.sum(section) != 0, f'Error: emptry section before interpolation for section {y0} '
                valid_coords_world, valid_coords_idx = get_valid_coords( coords, [y0w,y1w])

                if valid_coords_world.shape[0] != 0  :
                    x = np.rint( (valid_coords_world[:,0] - xstart)/xstep ).astype(int)
                    z = np.rint( (valid_coords_world[:,2] - zstart)/zstep ).astype(int)
                    xmin = np.min(x)
                    zmin = np.min(z)

                    xmax = np.max(x)
                    zmax = np.max(z)

                    #xz_valid_coords = (xmax < section.shape[0]) & (zmax < section.shape[1]) &  (zmin >= 0) & (xmin >= 0)
                    #valid_coords_idx = valid_coords_idx[xz_valid_coords] 
                    #x = x[xz_valid_coords] 
                    #z = z[xz_valid_coords] 

                    if zmax >= section.shape[1]: 
                        print(f'\n\nWARNING: z index {zmax} is greater than dimension {section.shape[1]}\n\n')
                    if xmax >= section.shape[0]: 
                        print(f'\n\nWARNING: x index {xmax} is greater than dimension {section.shape[0]}\n\n')
                    section = np.pad(section, (  (0, max(0,xmax-section.shape[0]+1)), 
                                                (0, max(0,zmax-section.shape[0]+1))))
                    # get nearest neighbour voxel intensities at x and z coordinate locations
                    values = section[x,z]
                    if np.sum(values>0) > 0: 'Error: empty section[x,z] in project_volume_to_surfaces'

                    assert np.sum(np.isnan(values)) == 0 , f'Error: nan found in values from {vol_fn}'
                    all_values[valid_coords_idx] = values 
            np.savetxt(interp_csv, all_values)

        assert np.sum(all_values>0) > 0, 'Error, empty array all_values in project_volumes_to_surfaces'

from scipy.interpolate import interp1d, CubicSpline

def setup_section_normalization(ligand, slab_df, array_src):
    normalize_sections = False 
    mean_list=[]
    std_list=[]
    y_list=[]
    group_mean=0
    group_std=0
    slab_df = slab_df.sort_values(['slab_order'])
    if ligand in [] :# [ 'cellbody' , 'myelin' ] : 
        print('Normalizing', ligand)

        normalize_sections=True  
        for row_i, row in slab_df.iterrows() : 
            y = int(row['slab_order'])

            # Conversion of radioactivity values to receptor density values
            section = array_src[:, y, :]
            
            idx = section>=0

            mean_list.append(np.mean(section[idx]))
            std_list.append(np.std(section[idx]))
            y_list.append(y)
        
        mean_list=np.array(mean_list) 
        pad=2
        new_mean_list=[]
        i_list = range(len(mean_list))
        
        new_mean_list.append(mean_list[0])
        for i, y, mean in zip(i_list[1:-1], y_list[1:-1], mean_list[1:-1]) :
            i0=max(i-pad,0)
            i1=min(i+pad,len(mean_list)-1)
            x = list(y_list[i0:i]) + list(y_list[i+1:i1+1])
            z = list(mean_list[i0:i]) + list(mean_list[i+1:i1+1])
            kind_dict={ 1:'nearest', 2:'linear',  3:'quadratic', 4:'cubic', 5:'cubic'}
            interp_f = interp1d(x,z,kind=kind_dict[min(5,len(x))])
            #interp_f = CubicSpline(x,z)
            new_mean = interp_f(y)
            new_mean_list.append(new_mean)
        new_mean_list.append(mean_list[-1])

        for y, new_mean in zip(y_list, new_mean_list):
            section = array_src[:, y, :]
            section[section>0] = new_mean + section[section>0] - np.mean(section[section>0]) #+ new_mean
            array_src[:,y,:] = section
        #plt.plot(mean_list,c='r')
        #plt.plot(new_mean_list,c='b')
        #plt.savefig('/tmp/tmp.png')
        #group_mean=np.mean(mean_list)
        #group_std = np.std(std_list)

    return array_src, normalize_sections

def thicken_sections(interp_dir, slab_dict, df_ligand, n_depths, resolution, tissue_type='' ):

    rec_thickened_dict = {} 
    target_file=f'nl_2d_vol{tissue_type}_fn'
    slab_dict_keys = slab_dict.keys() 
    slab_dict_keys = sorted(slab_dict_keys)
   
    ligand = np.unique(df_ligand['ligand'])[0]
    
    for  i in slab_dict_keys :
        slab = slab_dict[i]
        slab_df = df_ligand.loc[df_ligand['slab'].astype(int)==int(i)]
        source_image_fn = slab[target_file]

        print('\t\tSource image for thickening:', source_image_fn)
        thickened_fn = f'{interp_dir}thickened_{int(i)}_{ligand}_{resolution}{tissue_type}_l{n_depths}.nii.gz'

        print('\t\tThickened Filename', thickened_fn)
        
        if not os.path.exists(thickened_fn) :
            array_img = nib.load(source_image_fn)
            array_src = array_img.get_fdata()
            
            assert np.sum(array_src) != 0, 'Error: source volume for thickening sections is empty\n'+ source_image_fn


            array_src, normalize_sections = setup_section_normalization(ligand, slab_df, array_src)
            width = np.round(1*(1+resolution/(0.02*2))).astype(int)
            print('\t\tThickening sections to ', 0.02*width*2)
            dim=[array_src.shape[0], 1, array_src.shape[2]]
            rec_vol = np.zeros_like(array_src)

            for row_i, row in slab_df.iterrows() : 
                y = int(row['slab_order'])
                # Conversion of radioactivity values to receptor density values
                section = array_src[:, y, :].copy()

                if np.sum(section) == 0 : 
                    print(f'Warning: empty frame {i} {row}\n')
                
                section = section.reshape(dim)
                if row['conversion_factor'] > 0 and tissue_type != '_cls' :
                     section *= row['conversion_factor']
                elif row['conversion_factor'] == 0 : 
                    continue
                
                y0 = int(y)-width if int(y)-width > 0 else 0
                y1 = 1+int(y)+width if int(y)+width <= array_src.shape[1] else array_src.shape[1]
                #put ligand sections into rec_vol

                rep = np.repeat(section, y1-y0, axis=1)
                print('ligand', row['ligand'], row['slab_order'], np.max(section) ) 

                #rep = np.repeat(aa, y1-y0, axis=1)
                #rep[rep > 0] = aa[rep>0] #1000 # y0 #DEBUG 

                rec_vol[:, y0:y1, :] = rep 
            
            #if not normalize_sections : 
            #    assert np.sum(rec_vol) != 0, 'Error: receptor volume for single ligand is empty\n'
            nib.Nifti1Image(rec_vol, array_img.affine).to_filename(thickened_fn)

        rec_thickened_dict[i] = thickened_fn
    return rec_thickened_dict


def create_thickened_volumes(interp_dir, depth_list, depth_fn_slab_space, depth_fn_list, slab_dict, df_ligand, n_depths, resolution, origin=np.array([0,0,0]), tissue_type=''):
    for depth_index, (depth, depth_fn) in enumerate(zip(depth_list,depth_fn_list)):
        # Get surfaces transformed into slab space
        slabs = list(slab_dict.keys())
        slabs.sort()
        surf_fn_list = [ depth_fn_slab_space[i][depth] for i in slabs  ]
        
        # 1. Thicken sections
        thickened_dict = thicken_sections(interp_dir, slab_dict, df_ligand, n_depths, resolution, tissue_type=tissue_type)
        
        #thickened_dict = combat_slab_normalization(df_ligand, thickened_dict)

        # 2. Project autoradiograph densities onto surfaces
        print('\t\tProjecting volume to surface.')
        project_volumes_to_surfaces(surf_fn_list, thickened_dict, depth_fn, interp_dir, origin=origin)
    return thickened_dict

def get_profiles(profiles_fn, recon_out_prefix, depth_fn_mni_space, depth_list, depth_fn_list, nrows):
    print('\tGetting profiles')

    # 3. Interpolate missing densities over surface
    if not os.path.exists(profiles_fn) :
        
        profiles = h5.File(profiles_fn, 'w') 
        
        profiles.create_dataset('data', (nrows, len(depth_list)) )

        for depth_index, (depth, depth_fn) in enumerate(zip(depth_list, depth_fn_list)):
            print('\t\t\t\treading interpolated values from ',depth_fn)
            profiles_raw = pd.read_csv(depth_fn, header=None, index_col=None)
            assert np.sum(profiles_raw.values>0) > 0 , 'Error: empty depth file '+depth_fn

            sphere_rsl_fn = depth_fn_mni_space[depth]['sphere_rsl_fn'] 
            surface_val = profiles_raw.values.reshape(-1,) 
          
            profile_vector = interpolate_over_surface(sphere_rsl_fn, surface_val,threshold=0.02,order=1)

            profiles['data'][:,depth_index] = profile_vector
            del profile_vector

    return profiles_fn

def project_volume_to_surfaces(interp_dir, surf_dir, depth_list, slab_dict, df_ligand, depth_fn_mni_space, depth_fn_slab_space, n_depths, resolution, recon_out_prefix, tissue_type='', origin=np.array([0,0,0]), clobber=False):
    
    profiles_fn = f'{recon_out_prefix}_l{n_depths}_profiles.h5'
    
    example_depth_fn = depth_fn_mni_space[depth_list[0]]['upsample_fn']
    nrows = h5.File(example_depth_fn)['data'].shape[0]
    depth_fn_list = [ sub('.h5', f'_{depth}_l{n_depths}_raw.csv', profiles_fn) for depth in depth_list ]
   
    thickened_dict = create_thickened_volumes(interp_dir, depth_list, depth_fn_slab_space, depth_fn_list, slab_dict, df_ligand, n_depths, resolution, origin=origin, tissue_type=tissue_type)
    profiles_fn = get_profiles(profiles_fn, recon_out_prefix, depth_fn_mni_space, depth_list, depth_fn_list, nrows)

    return thickened_dict, profiles_fn
     
def interpolate_over_surface(sphere_obj_fn,surface_val,threshold=0,order=1):
    print('\tInterpolating Over Surface')
    print('\t\t\tSphere fn:',sphere_obj_fn)
    # get coordinates from dicitonary with mesh info
    coords = h5.File(sphere_obj_fn)['data'][:] 

    spherical_coords = surface_tools.spherical_np(coords) 

    #define a mask of verticies where we have receptor densitiies
    surface_mask = surface_val > threshold * np.max(surface_val)
    #a=1636763
    #b=1636762
    #print(coords[surface_mask.astype(bool)][a])
    #print(coords[surface_mask.astype(bool)][b])
    assert np.sum(surface_mask) != 0, "Error, empty profiles {}".format(np.sum(surface_mask))
    #define vector with receptor densities 
    surface_val_src = surface_val[ surface_mask.astype(bool) ]

    #define vector without receptor densities
    surface_val_tgt = surface_val[ ~surface_mask.astype(bool) ]

    # get coordinates for vertices in mask
    spherical_coords_src = spherical_coords[ surface_mask.astype(bool), : ]
    
    # get spherical coordinates from cortical mesh vertex coordinates

    #print(spherical_coords_src[a])
    #print(spherical_coords_src[b])
    lats_src, lons_src = spherical_coords_src[:,1]-np.pi/2, spherical_coords_src[:,2]
    #print(lats_src[a], lons_src[a])
    #print(lats_src[b], lons_src[b])

    temp = np.concatenate([(spherical_coords_src[:,1]-np.pi/2).reshape(-1,1), spherical_coords_src[:,2].reshape(-1,1)],axis=1)

    # create mesh data structure

    temp = np.concatenate([lons_src.reshape(-1,1),lats_src.reshape(-1,1)],axis=1)

    mesh = stripy.sTriangulation(lons_src, lats_src)
    lats, lons = spherical_coords[:,1]-np.pi/2, spherical_coords[:,2]
    
    interp_val, interp_type = mesh.interpolate(lons,lats, zdata=surface_val[surface_mask], order=order)
        
    return interp_val

def transform_surf_to_slab(interp_dir, slab_dict, depth_fn_space_mni, ref_gii_fn, ext, mni_fn, clobber=0):
    surf_rsl_dict={}
    for slab, cur_slab_dict in slab_dict.items() :
        surf_rsl_dict[slab]={}
        print(cur_slab_dict)
        for depth, depth_dict in depth_fn_space_mni.items() :
            upsample_fn = depth_dict['upsample_fn']
            upsample_gii_fn = depth_dict['upsample_gii_fn']
            sphere_rsl_fn = depth_dict['sphere_rsl_fn']
            surf_rsl_dict[slab][depth]={}
            upsample_slab_space_fn="{}/slab-{}_{}".format(interp_dir,slab,os.path.basename(upsample_fn))
            upsample_slab_space_gii="{}/slab-{}_{}".format(interp_dir,slab,os.path.basename(upsample_gii_fn))
            surf_rsl_dict[slab][depth]['upsample_h5'] = upsample_slab_space_fn
            surf_rsl_dict[slab][depth]['upsample_gii'] = upsample_slab_space_gii
             
            if not os.path.exists(upsample_slab_space_fn) or clobber >= 1 : 
                print(f"\t\tTransformig surface at depth {depth} to slab {slab}")
                apply_ants_transform_to_gii(upsample_fn, [cur_slab_dict['nl_3d_tfm_fn']], upsample_slab_space_fn, 0, depth_fn_space_mni[0]['faces_fn'], ref_gii_fn, ext)
     
    return surf_rsl_dict

class Ext():
    def __init__(self,gm_sphere,wm_sphere,gm,wm):
        self.gm_sphere = gm_sphere
        self.wm_sphere = wm_sphere
        self.gm = gm
        self.wm = wm

def get_surface_filename(surf_obj_fn, surf_gii_fn, surf_fs_fn):
    if os.path.exists(surf_gii_fn) :
        surf_fn = surf_gii_fn
        ext = Ext('.surf.gii', '.surf.gii', '.surf.gii', '.surf.gii')
    elif os.path.exists(surf_fs_fn):
        surf_fn = surf_fs_fn
        ext = Ext('.pial.sphere', '.white.sphere', '.pial', '.white')
    elif os.path.exists(surf_obj_fn):
        surf_fn = surf_gii_fn
        ext = Ext('.surf.gii', 'surf.gii', '.surf.gii', '.surf.gii')
        obj_to_gii(surf_obj_fn, ref_surf_fn, surf_fn)
    else :
        print('Error: could not find input GM surface (obj, fs, gii) for ', surf_gii_fn)
        exit(1)
    return surf_fn, ext

def prepare_surfaces(slab_dict, depth_list, interp_dir, resolution, upsample_resolution, mni_fn, surf_dir='civet/mri1/surfaces/surfaces/', n_vertices = 327696, brain='mri1', hemi='R', clobber=0):
    '''

    '''
    surf_rsl_dir = interp_dir +'/surfaces/' 
    os.makedirs(surf_rsl_dir, exist_ok=True)
    
    #Interpolate at coordinate locations
    ref_surf_fn = surf_base_str.format( surf_dir, brain,'gray', hemi, n_vertices,'','surf.gii')
    ref_surf_obj_fn = surf_base_str.format( surf_dir, brain,'gray', hemi, n_vertices,'','obj')

    #if not os.path.exists(ref_surf_fn) :
    #    shell('ConvertSurface -i_obj {ref_surf_obj_fn}  -o_gii {ref_surf_fn}')


    surf_gm_obj_fn = surf_base_str.format(surf_dir, brain, 'gray', hemi, n_vertices,'','obj')
    surf_wm_obj_fn = surf_base_str.format(surf_dir, brain, 'white', hemi, n_vertices,'','obj')
    
    surf_gm_gii_fn = surf_base_str.format(surf_dir, brain, 'gray', hemi, n_vertices,'','surf.gii')
    surf_wm_gii_fn = surf_base_str.format(surf_dir, brain, 'white', hemi, n_vertices,'','surf.gii')
   
    surf_gm_fs_fn = surf_base_str.format(surf_dir, brain, 'gray', hemi, n_vertices,'','pial')
    surf_wm_fs_fn = surf_base_str.format(surf_dir, brain, 'white', hemi, n_vertices,'','white')
    
    surf_gm_fn, ext = get_surface_filename(surf_gm_obj_fn, surf_gm_gii_fn, surf_gm_fs_fn)
    surf_wm_fn, _ = get_surface_filename(surf_wm_obj_fn, surf_wm_gii_fn, surf_wm_fs_fn)

    origin=[0,0,0]
    if ext.gm == '.pial' : origin= load_mesh(surf_gm_fn)[2]['cras']
    
    sphere_obj_fn = surf_base_str.format(surf_dir, brain, 'mid', hemi, n_vertices,'_sphere',ext.gm)
    
    #upsample transformed surfaces to given resolution
    print("\tUpsampling and inflating surfaces.") 
    depth_fn_mni_space = upsample_and_inflate_surfaces(surf_rsl_dir, surf_wm_fn, surf_gm_fn, ext, resolution, upsample_resolution, depth_list,slab_dict)

    #For each slab, transform the mesh surface to the receptor space
    #TODO: transform points into space of individual autoradiographs
    print("\tTransforming surfaces to slab space.") 
    depth_fn_slab_space = transform_surf_to_slab(surf_rsl_dir, slab_dict, depth_fn_mni_space, surf_wm_fn, ext.gm, mni_fn)

    return depth_fn_mni_space, depth_fn_slab_space, origin

class ImageParameters():
    def __init__(self, starts, steps, dimensions):
        self.starts = starts
        self.steps = steps 
        self.dimensions = dimensions
        self.affine = np.array([[self.steps[0], 0, 0, self.starts[0]],
                       [0, self.steps[1], 0, self.starts[1]],
                       [0, 0, self.steps[2], self.starts[2]],
                       [0, 0, 0, 1]])
        #print('\tStarts', self.starts)
        #print('\tSteps', self.steps)
        #print('\tDimensions:', self.dimensions)
        #print('\tAffine', self.affine)


def get_image_parameters(fn ):
    img = nib.load(fn)
    affine = nb_surf.load(fn).affine

    starts = np.array(affine[[0,1,2],3])
    steps = affine[[0,1,2],[0,1,2]]

    dimensions = img.shape

    imageParam = ImageParameters(starts, steps, dimensions)

    return imageParam

def combine_interpolated_sections(slab_fn, interp_vol, ystep_lo, ystart_lo) :
    img = nib.load(slab_fn)
    vol = img.get_fdata()

    affine = nb_surf.load(slab_fn).affine
    ystep_hires = affine[1,1]
    ystart_hires = affine[1,3]

    print('\tCombining interpolated sections with original sections')

    for y in range(vol.shape[1]) :
        
        section_is_empty = np.sum(vol[:,y,:]) <= 0
        
        # convert from y from thickened filename 
        yw = y * ystep_hires + ystart_hires
        y_lo = np.rint( (yw - ystart_lo)/ystep_lo).astype(int)

        if not section_is_empty and y_lo < interp_vol.shape[1] :
            #use if using mapper interpolation interp_section = interp_vol[ : , yi, : ]
            original_section = vol[ : , y, : ]

            interp_vol[:, y_lo, :] = original_section 
        else :
            print('Not including section', np.sum(vol[:,y,:]))

    return interp_vol


def create_reconstructed_volume(interp_fn_list, interp_dir, thickened_fn_dict, profiles_fn, depth_list, depth_fn_slab_space, slabs, files, resolution, df_ligand, scale_factors_json, use_mapper=True, clobber=False, origin=[0,0,0],gm_label = 2. ):
            
    profiles = None
    print(scale_factors_json) 
    for interp_fn, slab in zip(interp_fn_list, slabs) :
        print(slab, type(slab))
        sectioning_direction =  scale_factors_json[slab]['direction']
        df_ligand_slab = df_ligand.loc[ df_ligand['slab'].astype(int) == int(slab) ]

        if not os.path.exists(interp_fn) or clobber : 
            print('\tReading profiles', profiles_fn) 
            if type(profiles) != type(np.array) : profiles = h5.File(profiles_fn, 'r')['data'][:]
            assert np.sum(profiles>0) > 0, 'Error: profiles h5 is empty: '+profiles_fn
            #profiles_bin = np.copy(profiles)
            #profiles_bin[ profiles_bin > 0 ] = 1 
            # Hiad dimensions for output volume
            files_resolution = files[str(int(slab))]
            resolution_list = list(files_resolution.keys())
            max_resolution = resolution_list[-1]
            slab_fn = files_resolution[ max_resolution ]['nl_2d_vol_fn'] 
            srv_space_rec_fn = files_resolution[ max_resolution ]['srv_space_rec_fn']
            srv_iso_space_rec_fn = files_resolution[ max_resolution ]['srv_iso_space_rec_fn']
            thickened_fn = thickened_fn_dict[ slab ]
            print('Thickened Filename', thickened_fn)
            
            #out_affine= nib.load(slab_fn).affine
            out_affine= nib.load(srv_iso_space_rec_fn).affine

            imageParamHi = get_image_parameters(slab_fn)
            imageParamLo = get_image_parameters(srv_iso_space_rec_fn) 
            ''' 
            #Create an object that will be used to interpolate over the surfaces
            wm_upsample_fn = depth_fn_slab_space[slab][depth_list[0]]['upsample_gii']
            gm_upsample_fn = depth_fn_slab_space[slab][depth_list[-1]]['upsample_gii']
            npz_dir = interp_dir+f'/npz-{slab}_{resolution}mm/'
            os.makedirs(npz_dir, exist_ok=True)
            mapper = SurfaceVolumeMapper(white_surf=wm_upsample_fn, gray_surf=gm_upsample_fn, 
                        resolution=imageParamHi.steps, dimensions=imageParamHi.dimensions, 
                        origin=imageParamHi.starts, filename=None, save_in_absence=False, mask=None,
                        out_dir=npz_dir, left_oriented=False )
            
            mask_vol = mapper.map_profiles_to_block(profiles=profiles_bin, interpolation='nearest')
            '''
            print(srv_iso_space_rec_fn)
            mask_img=nib.load(srv_iso_space_rec_fn)
            mask_vol = mask_img.get_fdata()

            gm_lo = gm_label * 0.75
            gm_hi = gm_label * 1.25
            valid_idx = (mask_vol >= gm_lo) & (mask_vol < gm_hi)
            mask_vol = np.zeros_like(mask_vol).astype(np.uint8)
            mask_vol[ valid_idx ] = 1
            
            # Use algorithm that averages vertex values into voxels
            interp_only_fn = re.sub('.nii','_interp-only.nii', interp_fn)
            multi_mesh_interp_fn = re.sub('.nii','_multimesh.nii', interp_fn)
            gm_mask_fn = re.sub('.nii','_gm-mask.nii', interp_fn)
            nib.Nifti1Image(mask_vol, out_affine ).to_filename(gm_mask_fn)

            if not os.path.exists(multi_mesh_interp_fn) : 
                interp_vol = multi_mesh_to_volume(profiles, depth_fn_slab_space[slab], depth_list, imageParamLo.dimensions, imageParamLo.starts, imageParamLo.steps, origin=origin)
                nib.Nifti1Image(interp_vol, out_affine ).to_filename(multi_mesh_interp_fn)
                interp_vol = fill_in_missing_voxels(interp_vol, mask_vol)
            else : 
                interp_vol = nib.load(multi_mesh_interp_fn).get_fdata()

            #if sectioning_direction  == 'rostral_to_caudal' :
            #    limit = df_ligand_slab['slab_order'].max()
            #    interp_vol[ :, int(limit):, : ] = 0
            #else :
            #    limit = df_ligand_slab['slab_order'].min()
            #    print('LIMIT', limit, sectioning_direction)
            #    print('INTERP VOL SUM', np.sum(interp_vol))
            #    
            #    interp_vol[ :, 0:int(limit), : ] = 0
            #    print('INTERP VOL SUM', np.sum(interp_vol))
           
            ystep_interp = imageParamHi.steps[1]
            ystart_interp = imageParamHi.starts[1]

            assert np.sum(interp_vol) > 0 , f'Error: interpolated volume is empty using {profiles_fn}'

            df_ligand_slab = df_ligand.loc[ df_ligand['slab'].astype(int)==int(slab)]
            print('Writing', interp_only_fn)
            nib.Nifti1Image(interp_vol, mask_img.affine ).to_filename(interp_only_fn)

            output_vol = combine_interpolated_sections(thickened_fn, interp_vol, imageParamLo.steps[1], imageParamLo.starts[1])
            #DEBUG
            #output_vol = interp_vol

            # prefilter the output volume for when it gets resampled to mni space
            print('Writing', interp_fn)
            nib.Nifti1Image(output_vol, mask_img.affine ).to_filename(interp_fn)
            print('\nDone.')
        else :
            print(interp_fn, 'already exists')


def fill_in_missing_voxels(interp_vol, mask_vol):
    print('\tFilling in missing voxels.')
    mask_vol = np.rint(mask_vol)
    mask_vol = np.pad(mask_vol, ((1,1),(1,1),(1,1)) )
    interp_vol = np.pad(interp_vol, ((1,1),(1,1),(1,1)) )

    xv, yv, zv = np.meshgrid(np.arange(mask_vol.shape[0]), 
                                np.arange(mask_vol.shape[1]), 
                                np.arange(mask_vol.shape[2]))

    xv = xv.reshape(-1,1)
    yv = yv.reshape(-1,1)
    zv = zv.reshape(-1,1)

    missing_voxels = (mask_vol[xv,yv,zv] > 0) & (interp_vol[xv,yv,zv] == 0)
    
    counter=0
    last_missing_voxels = np.sum(missing_voxels) + 1
    print( np.sum(missing_voxels) > 0 , np.sum(missing_voxels) < last_missing_voxels )
    while np.sum(missing_voxels) > 0 and np.sum(missing_voxels) < last_missing_voxels: 
        last_missing_voxels = np.sum(missing_voxels)
        xvv = xv[missing_voxels]
        yvv = yv[missing_voxels]
        zvv = zv[missing_voxels]
        
        xvp = xvv + 1
        xvm = xvv - 1
        yvp = yvv + 1
        yvm = yvv - 1
        zvp = zvv + 1
        zvm = zvv - 1

        
        x0 = np.vstack([xvp, yvv, zvv]).T
        x1 = np.vstack([xvm, yvv, zvv]).T
        y0 = np.vstack([xvv, yvp, zvv]).T
        y1 = np.vstack([xvv, yvm, zvv]).T
        z0 = np.vstack([xvv, yvv, zvm]).T
        z1 = np.vstack([xvv, yvv, zvp]).T
       


        interp_values = np.vstack( [interp_vol[ x0[:,0],x0[:,1],x0[:,2]],
                                    interp_vol[ x1[:,0],x1[:,1],x1[:,2]], 
                                    interp_vol[ y0[:,0],y0[:,1],y0[:,2]], 
                                    interp_vol[ y1[:,0],y1[:,1],y1[:,2]],
                                    interp_vol[ z0[:,0],z0[:,1],z0[:,2]],
                                    interp_vol[ z1[:,0],z1[:,1],z1[:,2]]]).T
       
        n = np.sum(interp_values > 0, axis=1)

        interp_sum = np.sum(interp_values, axis=1)

        xvv = xvv[n>0]
        yvv = yvv[n>0]
        zvv = zvv[n>0]

        interp_values = interp_sum[n>0] / n[n>0]
   
        interp_vol[xvv, yvv, zvv] = interp_values
  
        missing_voxels = (mask_vol[xv,yv,zv] > 0 ) & (interp_vol[xv,yv,zv] == 0)
        print('missing', np.sum(missing_voxels))
        counter += 1

    interp_vol = interp_vol[1:-1,1:-1,1:-1]
    return interp_vol


def interpolate_between_slabs(depth_fn_mni_space, depth_list, profiles_fn, ref_fn, interp_dir, srv_rsl_fn, resolution, origin=[0,0,0]):
    img = nb_surf.load(ref_fn)
    starts = np.array(img.affine[[0,1,2],3])
    steps = np.array(img.affine[[0,1,2],[0,1,2]])
    dimensions = img.shape

    wm_upsample_fn = depth_fn_mni_space[depth_list[0]]['upsample_gii_fn']
    gm_upsample_fn = depth_fn_mni_space[depth_list[-1]]['upsample_gii_fn']
    print('\tReading:', profiles_fn)
    profiles = h5.File(profiles_fn, 'r')['data'][:]
    mask_vol = np.rint(nib.load(srv_rsl_fn).get_fdata() )
   
    interp_vol = multi_mesh_to_volume(profiles, depth_fn_mni_space, depth_list, dimensions, starts, steps, origin=origin)
    mask_vol = resize(mask_vol, interp_vol.shape, order=0) 
    
    interp_vol = fill_in_missing_voxels(interp_vol, mask_vol)
    return interp_vol, mask_vol

def combine_slabs_to_volume(interp_fn_mni_list, output_fn):
    ref_img = nib.load(interp_fn_mni_list[0])
    output_volume = np.zeros(ref_img.shape)

    if not os.path.exists(output_fn) :
        n= np.zeros_like(output_volume)
        for interp_mni_fn in interp_fn_mni_list :
            vol = nib.load(interp_mni_fn).get_fdata()
            output_volume += vol
            n[ vol > 0 ] += 1
        #FIXME taking the average can attenuate values in overlapping areas
        output_volume[n>0] /= n[n>0]
        output_volume[ np.isnan(output_volume) ] = 0
        nib.Nifti1Image(output_volume, ref_img.affine, direction_order='lpi' ).to_filename( output_fn )
    
    return output_fn


def find_ligands_to_reconstruction(slabs, interp_dir, template_out_prefix ):
    interp_fn_list = []
    interp_fn_mni_list = []
    for slab in slabs :
        slab_out_prefix = sub('_slab_', f'_{slab}_', template_out_prefix)
        interp_fn  = f'{slab_out_prefix}_space-slab.nii.gz'
        interp_space_mni_fn  = f'{slab_out_prefix}_space-mni.nii.gz'
        interp_fn_list.append( interp_fn ) 
        interp_fn_mni_list.append(interp_space_mni_fn)

    return interp_fn_list, interp_fn_mni_list

def transform_slab_to_mni(slabs, thickened_dict, slab_dict, mni_fn,  template_out_prefix):

    for slab in slabs :
        slab_out_prefix = sub('_slab_', f'_{slab}_', template_out_prefix)
        #DEBUG
        #interp_fn  = f'{slab_out_prefix}_space-slab.nii.gz'
        interp_fn = thickened_dict[slab]
        interp_space_mni_fn  = f'{slab_out_prefix}_space-mni.nii.gz'
        tfm = slab_dict[slab]['nl_3d_tfm_fn']
        if not os.path.exists(interp_space_mni_fn) :
            shell(f'antsApplyTransforms -d 3 -n NearestNeighbor -i {interp_fn} -r {mni_fn} -t {tfm} -o {interp_space_mni_fn}')

def create_final_reconstructed_volume(final_mni_fn, mni_fn, resolution,  depth_fn_mni_space, depth_list, interp_dir, interp_fn_mni_list, recon_out_prefix, profiles_fn, n_depths, origin=[0,0,0] ):
    combined_slab_mni_fn = f'{recon_out_prefix}_space-mni_not_filled_l{n_depths}.nii.gz'
    surf_interp_mni_fn = f'{recon_out_prefix}_surf-interp_space-mni_l{n_depths}.nii.gz'
    final_mask_fn = f'{recon_out_prefix}_surf-interp_space-mni_l{n_depths}_mask.nii.gz'

    ref_img = nib.load(mni_fn) 
    ystart = ref_img.affine[1,3]
    combine_slabs_to_volume(interp_fn_mni_list, combined_slab_mni_fn)

    print('\tInterpolate between slabs')
    if not os.path.exists(surf_interp_mni_fn):
        interp_vol, mask_vol = interpolate_between_slabs(depth_fn_mni_space, depth_list, profiles_fn, interp_fn_mni_list[0], interp_dir, mni_fn,  resolution, origin=origin)
    
        #print('\tWriting surface interpolation volume:\n\t\t', final_mni_fn)
        nib.Nifti1Image(interp_vol, ref_img.affine, direction_order='lpi').to_filename(surf_interp_mni_fn)
    
    # fill in missing sections in whole brain
    print('\tCombine interpoalted section with slabs to fill gaps')
    combined_slab_vol = nib.load(combined_slab_mni_fn).get_fdata()
    #FIXME DEBUG
    #idx = combined_slab_vol <=300
    idx = combined_slab_vol < 0.02 * np.max(combined_slab_vol)
    
    combined_slab_vol[ idx ] = interp_vol[ idx ]
    output_vol = combined_slab_vol
    
    output_vol = interp_vol
    print('\tWriting', final_mni_fn)
    nib.Nifti1Image(output_vol, ref_img.affine, direction_order='lpi').to_filename(final_mni_fn)
    print('\tWriting', final_mask_fn)
    nib.Nifti1Image(mask_vol, ref_img.affine, direction_order='lpi').to_filename(final_mask_fn)


def surface_interpolation(df_ligand, slab_dict, out_dir, interp_dir, brain, hemi, resolution, mni_fn, slabs, files, scale_factors_json, n_depths=3, upsample_resolution=0, tissue_type='', surf_dir='civet/mri1/surfaces/surfaces/', n_vertices = 327696, gm_label=2, clobber=0):
    if upsample_resolution == 0 : upsample_resolution=resolution

    ligand = df_ligand['ligand'].values[0]

    template_out_prefix=f'{interp_dir}/{brain}_{hemi}_slab_{ligand}_{resolution}mm_l{n_depths}{tissue_type}'
    recon_out_prefix=f'{interp_dir}/{brain}_{hemi}_{ligand}_{resolution}mm{tissue_type}'
    final_mni_fn = f'{recon_out_prefix}_space-mni_l{n_depths}.nii.gz'

    interp_fn_list, interp_fn_mni_list = find_ligands_to_reconstruction(slabs, interp_dir, template_out_prefix)

    required_outputs = [ final_mni_fn ] + interp_fn_list + interp_fn_mni_list

    if  False in [ os.path.exists(out_fn) for out_fn in required_outputs]:
        
        print('\tReconstructing', ligand)

        #make sure resolution is interpreted as float
        resolution=float(resolution) 
        
        os.makedirs(interp_dir, exist_ok=True)

        #set depths
        dt = 1.0/ n_depths
        depth_list = np.arange(0, 1+dt/10, dt)
        depth_list = np.insert(depth_list,0, 0)
        print('\tPreparing surfaces for surface-interpolation')
        #FIXME brain as defined in surface files might be different (lowercase vs caps) than in dataframe
        depth_fn_mni_space, depth_fn_slab_space, origin = prepare_surfaces(slab_dict, depth_list, interp_dir, resolution, upsample_resolution, mni_fn, surf_dir=surf_dir, n_vertices = n_vertices, brain=brain, hemi=hemi, clobber=clobber)

        print('\tInterpolating for ligand:',ligand)

        # Extract profiles from the slabs using the surfaces 
        thickened_dict, profiles_fn = project_volume_to_surfaces(interp_dir, interp_dir+'/surfaces/', depth_list, slab_dict, df_ligand, depth_fn_mni_space, depth_fn_slab_space, n_depths, resolution,  recon_out_prefix,origin=origin, tissue_type=tissue_type)
        
        # Interpolate a 3D receptor volume from the surface mesh profiles
        print('\tCreate Reconstructed Volume')
        create_reconstructed_volume(interp_fn_list, interp_dir, thickened_dict, profiles_fn, depth_list, depth_fn_slab_space, slabs, files, resolution, df_ligand, scale_factors_json, use_mapper=True, clobber=clobber, origin=origin, gm_label=gm_label)
        
        # transform interp_fn to mni space
        print('\tTransform slab to mni')
        transform_slab_to_mni(slabs, thickened_dict, slab_dict,mni_fn, template_out_prefix)
        
        # interpolate from surface to volume over entire brain
        print('\tCreate final reconstructed volume')
        create_final_reconstructed_volume(final_mni_fn, mni_fn, resolution, depth_fn_mni_space, depth_list, interp_dir, interp_fn_mni_list, recon_out_prefix, profiles_fn, n_depths, origin=origin )

    return final_mni_fn

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--clobber', dest='clobber', type=int, default=0, help='Clobber results')
    parser.add_argument('--brain', dest='brain', type=str, help='brain')
    parser.add_argument('--hemi', dest='hemi', type=str,  help='hemi')
    parser.add_argument('--out-dir', dest='out_dir', type=str,  help='Clobber results')
    parser.add_argument('--lin-df-fn', dest='lin_df_fn', type=str,  help='Clobber results')
    parser.add_argument('--slab-str', dest='slab_str', type=str,  help='Clobber results')
    parser.add_argument('--n-depths', dest='n_depths', type=int,  default=8, help='Clobber results')
    args = parser.parse_args()
    
    lin_df_fn = args.lin_df_fn
    out_dir = args.out_dir
    brain = args.brain
    hemi = args.hemi
    n_depths = args.n_depths
    slab_str = args.slab_str
    clobber=args.clobber
