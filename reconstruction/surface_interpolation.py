import os
import argparse
import utils.ants_nibabel as nib
#import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py as h5
import pandas as pd
import stripy as stripy
import numpy as np
import vast.surface_tools as surface_tools
import ants
import tempfile
import time
from utils.utils import get_section_intervals
from utils.combat_slab_normalization import combat_slab_normalization
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label
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
from upsample_gifti import *
from vast.surface_volume_mapper import SurfaceVolumeMapper
#from pykrige.ok import OrdinaryKriging

global surf_base_str
surf_base_str = '{}/mri1_{}_surface_right_{}{}.{}'
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
 
def apply_ants_transform_to_gii( in_gii_fn, tfm_list, out_gii_fn, invert, faces_fn, ref_gii_fn):
    print("transforming", in_gii_fn)
    #faces, coords = nib.load(in_gii_fn).agg_data(('triangle', 'pointset'))
    coords = h5py.File(in_gii_fn)['data'][:]
    tfm = ants.read_transform(tfm_list[0])
    flip = 1
    if np.sum(tfm.fixed_parameters) != 0 : flip=-1
    
    in_file = open(in_gii_fn, 'r')
    
    out_path, out_ext = os.path.splitext(out_gii_fn)
    coord_fn = out_path + '_ants_reformat.csv'

    #read the csv with transformed vertex points
    with open(coord_fn, 'w+') as f :  
        f.write('x,y,z,t,label\n') 
        for x,y,z in coords :  
            f.write('{},{},{},{},{}\n'.format(flip*x,flip*y,z,0,0 ))
            #not zyx

    temp_out_fn=tempfile.NamedTemporaryFile().name+'.csv'
    shell(f'antsApplyTransformsToPoints -d 3 -i {coord_fn} -t [{tfm_list[0]},{invert[0]}]  -o {temp_out_fn}',verbose=True)

    # save transformed surfaced as an gii file
    with open(temp_out_fn, 'r') as f :
        #read the csv with transformed vertex points
        for i, l in enumerate( f.readlines() ):
            if i == 0 : continue
            x,y,z,a,b = l.rstrip().split(',')
            coords[i-1] = [flip*float(x),flip*float(y),float(z)]
    
    f_h5 = h5py.File(out_gii_fn, 'w')
    f_h5.create_dataset('data', data=coords) 

    faces = h5py.File(faces_fn,'r')['data'][:]
    print('ref gii fn',ref_gii_fn)
    print('out gii', out_path+'.surf.gii')
    save_gii(coords,faces,ref_gii_fn,out_path+'.surf.gii')
    obj_fn = out_path+ '.obj'
    save_obj(obj_fn,coords,faces)

def upsample_and_inflate_surfaces(surf_dir, wm_surf_fn, gm_surf_fn, resolution,  depth_list, clobber=False, n_vertices=81920):
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

    wm_coords = wm_mesh.agg_data('NIFTI_INTENT_POINTSET')
    wm_faces =  wm_mesh.agg_data('NIFTI_INTENT_TRIANGLE')
    
    #gm_upsample_fn="{}/surf_{}mm_{}_rsl.surf.gii".format(surf_dir,resolution,0)
    gm_obj_fn="{}/surf_{}mm_{}_rsl.obj".format(surf_dir,resolution,0)
    wm_upsample_fn="{}/surf_{}mm_{}_rsl.surf.gii".format(surf_dir,resolution,1)
    gm_sphere_fn = "{}/surf_{}mm_{}_inflate.surf.gii".format(surf_dir,resolution,0)
    gm_sphere_rsl_fn = "{}/surf_{}mm_{}_inflate_rsl.surf.gii".format(surf_dir,resolution,0)
    
    d_coords = wm_coords - gm_coords 
    del wm_coords
    
    input_list = []
    output_list = []

    upsample_0_fn = "{}/surf_{}mm_{}_rsl.surf.gii".format(surf_dir,resolution,depth_list[0])
    upsample_1_fn = "{}/surf_{}mm_{}_rsl.surf.gii".format(surf_dir,resolution,depth_list[-1])

    for depth in depth_list :
        print("\tDepth", depth)
        depth_surf_fn = "{}/surf_{}mm_{}.surf.gii".format(surf_dir,resolution,depth)
        upsample_fn = "{}/surf_{}mm_{}_rsl.h5".format(surf_dir,resolution,depth)
        upsample_gii_fn = "{}/surf_{}mm_{}_rsl.surf.gii".format(surf_dir,resolution,depth)
        sphere_fn = "{}/surf_{}mm_{}_inflate.surf.gii".format(surf_dir,resolution,depth)
        sphere_rsl_fn = "{}/surf_{}mm_{}_inflate_rsl.h5".format(surf_dir,resolution,depth)
        depth_fn_dict[depth]={'upsample_fn':upsample_fn, 'upsample_gii_fn':upsample_gii_fn, 'sphere_rsl_fn':sphere_rsl_fn}
        
        coords = gm_coords + depth * d_coords
        
        if not os.path.exists(depth_surf_fn) :
            save_gii( coords, wm_faces, gm_surf_fn, depth_surf_fn)

        del coords
   
        if not os.path.exists(sphere_fn) or  clobber :
            print('\tInflate to sphere')
            shell('~/freesurfer/bin/mris_inflate -n 500  {} {}'.format(depth_surf_fn, sphere_fn))
        
        input_list += [depth_surf_fn, sphere_fn]
        output_list+= [upsample_fn, sphere_rsl_fn]

    depth_fn_dict[depth_list[0]]['upsample_gii_fn'] = upsample_0_fn
    depth_fn_dict[depth_list[-1]]['upsample_gii_fn'] = upsample_1_fn

    faces_fn, coords_fn = upsample_gifti(gm_surf_fn, upsample_0_fn, upsample_1_fn, float(resolution), input_list=input_list, output_list=output_list, clobber=clobber)

    depth_fn_dict[depth_list[0]]['faces_fn'] = faces_fn

    rsl_faces = h5py.File(faces_fn,'r')['data'][:]
    rsl_coords = h5py.File(coords_fn, 'r')['data'][:]
    save_obj(gm_obj_fn, rsl_coords,rsl_faces)

    return depth_fn_dict



def load_slabs(slab_fn_list) :
    slabs=[]
    for slab_fn in slab_fn_list :
        slabs.append( nib.load(slab_fn) )
    return slabs

def vol_surf_interp(val, src, coords, affine, clobber=0 ):
    steps = [ affine[0,0], affine[1,1], affine[2,2]   ]
    starts =  [ affine[0,3], affine[1,3], affine[2,3]   ] 
    xmax=starts[0] + src.shape[0] * steps[0]
    ymax=starts[1] + src.shape[1] * steps[1]
    zmax=starts[2] + src.shape[2] * steps[2]
    
    vx = np.array([ w2v(i, steps[0], starts[0]) for i in coords[:,0]]).astype(int)
    vy = np.array([ w2v(i, steps[1], starts[1]) for i in coords[:,1]]).astype(int)
    vz = np.array([ w2v(i, steps[2], starts[2]) for i in coords[:,2]]).astype(int)
    
    idx = (vx >= 0) & (vy >= 0) & (vz >= 0) & (vx < src.shape[0]) & ( vy < src.shape[1]) & ( vz < src.shape[2] )
    assert np.sum(idx) != 0 , 'Error: no voxels found to interpolate over'

    val[idx] = src[vx[idx], vy[idx], vz[idx]]#.reshape(-1,)
    
    assert np.sum(val) != 0, "Error: no surface values projected onto surface"
    return val



    

def get_valid_coords( coords, iw):

    valid_coords_idx = (coords[:,1] >= iw[0]) & (coords[:,1]<iw[1])
    valid_coords_idx = valid_coords_idx.reshape(valid_coords_idx.shape[0])
    valid_coords = coords[valid_coords_idx, :]

    return valid_coords, valid_coords_idx

#def voxel_to_mesh_nearest(volume, coords, step, start):

def voxel_to_mesh(volume, coords, step, start):
    '''
    About :
        Performs trilinear interpolation of voxel values onto a surface mesh

    Arguments :
        volume

    Return :
        vertex_values
    '''

    #create array of index values for each of the coordinates
    coords_index = np.arange(coords.shape[0]).astype(int)

    # Convert the coordinates of mesh points to temporary voxel locations
    # These values must be floating point so that they can be then rounded up and down to find
    # voxel coordinates surrounding the mesh point
    temp_x_voxel = ((coords[:,0] - start[0]) / step[0]).reshape(-1,1)
    temp_y_voxel = ((coords[:,1] - start[1]) / step[1]).reshape(-1,1)
    temp_z_voxel = ((coords[:,2] - start[2]) / step[2]).reshape(-1,1)
    # Round the temporary voxel locations up or down to find the voxel neighbours
    x_voxel = np.hstack( [ np.floor(temp_x_voxel), np.ceil(temp_x_voxel) ] ).astype(int) 
    y_voxel = np.hstack( [ np.floor(temp_y_voxel), np.ceil(temp_y_voxel) ] ).astype(int)
    z_voxel = np.hstack( [ np.floor(temp_z_voxel), np.ceil(temp_z_voxel) ] ).astype(int)

    voxel_list = [ x_voxel, y_voxel, z_voxel ]
    for dim in range(3) :
        assert np.max(voxel_list) < volume.shape[dim], 'Error: coords larger in dimension {dim} ({np.max(voxel_list[dim])}) is too large compared to volume with dimensions {volume.shape}' 

    del temp_x_voxel
    del temp_y_voxel
    del temp_z_voxel

    v2w_vector = lambda vector, step, start : (vector * step - start + step/2.).reshape(-1,1)

    # Convert the neighbouring voxel locations to world coordinates
    x_world = np.hstack( [ v2w_vector(x_voxel[:,0], step[0], start[0]), v2w_vector(x_voxel[:,1], step[0], start[0]) ] )
    y_world = np.hstack( [ v2w_vector(y_voxel[:,0], step[1], start[1]), v2w_vector(y_voxel[:,1], step[1], start[1]) ] ) 
    z_world = np.hstack( [ v2w_vector(z_voxel[:,0], step[2], start[2]), v2w_vector(z_voxel[:,1], step[2], start[2]) ] ) 

    # Combine the x, y, and z coordinates into the 8 possible voxel points around each mesh point
    ngh_coords_world = np.array([])
    ngh_coords_voxel = np.array([])

    def aggregate_vectors(vector0, vector1, vector2, aggregated_vector):

        temp = np.hstack([vector0.reshape(-1,1), vector1.reshape(-1,1), vector2.reshape(-1,1)])

        if aggregated_vector.shape == (0,) :
            aggregated_vector = temp.astype(int)
        else :
            aggregated_vector = np.append( aggregated_vector, temp, axis=0).astype(int)

        return aggregated_vector

    for i in range(2):
        for j in range(2) :
            for k in range(2):

                ngh_coords_world = aggregate_vectors(x_world[:,i], y_world[:,j], z_world[:,k], ngh_coords_world)

                ngh_coords_voxel = aggregate_vectors(x_voxel[:,i], y_voxel[:,j], z_voxel[:,k], ngh_coords_voxel)

    # Repeat the coordiantes 8 times, one for each of the neighbouring voxel points to which they will be 
    # compared to calculate distances between mesh points and neighbouring voxel points
    coords_repeated = np.repeat([coords], 8, axis=0).reshape(-1,3)
    coords_index_repeated = np.repeat([coords_index], 8, axis=0).reshape(-1,).astype(int)
    del coords_index
    

    # Calculate distances between mesh points and world coordinates
    distances = np.sqrt(np.sum(np.power(coords_repeated - ngh_coords_world, 2),axis=1))

    distances[distances==0] = 1
    p=2
    weights = 1/np.power(distances,p)


    #for cr, ncw,d in zip(coords_repeated, ngh_coords_world, distances) :
    #    if d > 1 : 
    #        print(cr, ncw, d)

    #assert np.max(distances) <= np.max(np.sqrt(step**2+step**2)), f'Error: max distance between vertex and neighbouring voxels ({np.max(distances)}) is greater than maximum step size ({np.max(step)})'

    # Calculate the total distances around each coordinate using bincount
    weights_sum = np.bincount(coords_index_repeated, weights = weights)

    # because there are as many total distances as coordinates, we need to repeat the 
    # total distances 8 times to use them to weight the mesh point - voxel distances
        
    # Distances are weighted by total weights
    #weighted_distances =  distances / total_distances_repeated

    # Get the voxel values for all the neighbouring voxels
    voxel_values = volume[ ngh_coords_voxel[:,0], ngh_coords_voxel[:,1], ngh_coords_voxel[:,2] ]
    del ngh_coords_voxel

    
    for i in [10,100,999] :
        idx = coords_index_repeated == i
        for a,b,c,d,e,f in zip( coords_repeated[idx], ngh_coords_world[idx], voxel_values[idx], distances[idx], weights[idx], voxel_values[idx]  ):
            print(i,'\t',a,b,c,d,e,f)
    
    # Calculate 
    weighted_voxel_values = voxel_values  * weights
    vertex_values = np.bincount( coords_index_repeated, weights = weighted_voxel_values )
    vertex_values /= weights_sum

    del coords_index_repeated

    
    return vertex_values


def mesh_to_volume(coords, vertex_values, dimensions, starts, steps, vol_interp=None, n_vol=None ):
    '''
    About
        Interpolate mesh values into a volume
    Arguments
        coords
        vertex_values
        dimensions
        starts
        steps
        vol_interp
        n_vol
    Return
        vol_interp
        n_vol
    '''
    if type(vertex_values) != np.ndarray  or type(n_vol) != np.ndarray :
        vol_interp = np.zeros(dimensions)
        n_vol = np.zeros_like(vol_interp)

    x = np.rint( (coords[:,0] - starts[0]) / steps[0] ).astype(int)
    y = np.rint( (coords[:,1] - starts[1]) / steps[1] ).astype(int)
    z = np.rint( (coords[:,2] - starts[2]) / steps[2] ).astype(int)

    idx = (x >= 0) & (y >= 0) & (z >= 0) & (x < dimensions[0]) & ( y < dimensions[1]) & ( z < dimensions[2] )
    x = x[idx]
    y = y[idx]
    z = z[idx]
    #for c, i, j, k, v in zip(coords,x,y,z,vertex_values):

    vol_interp[x,y,z] +=  vertex_values[idx]

    n_vol[x,y,z] += 1

    return vol_interp, n_vol

def multi_mesh_to_volume(profiles, depth_fn_slab_space, depth_list, dimensions, starts, steps):
    
    vol_interp = np.zeros(dimensions)
    n_vol = np.zeros_like(vol_interp)


    for ii in range(profiles.shape[1]):
        surf_fn = depth_fn_slab_space[depth_list[ii]]['upsample_h5']
        coords = h5py.File(surf_fn,'r')['data'][:]
        vol_interp, n_vol = mesh_to_volume(coords, profiles[:,ii], dimensions, starts, steps, vol_interp, n_vol )

    vol_interp[ n_vol>0 ] = vol_interp[n_vol>0] / n_vol[n_vol>0]
    
    assert np.sum(vol_interp) != 0 , 'Error: interpolated volume is empty'
    return vol_interp

def project_volumes_to_surfaces(surf_fn_list, thickened_dict, interp_csv, interp_dir, clobber=False):
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

    if not os.path.exists(interp_csv) or clobber: 
        qc_dir = interp_dir + '/qc/'
        os.makedirs(qc_dir, exist_ok=True)
        
        nvertices = h5py.File(surf_fn_list[0]['upsample_h5'],'r')['data'].shape[0]
        #the default value of the vertices is set to -100 to make it easy to distinguish
        #between vertices where a ligand density of 0 is measured versus the default value
        all_values=np.zeros(nvertices) - 100
    
        print(thickened_dict);
        #Iterate over slabs within a given 
        for i, surf_fn in enumerate(surf_fn_list) :
            print(i)
            print(thickened_dict.keys())
            print(thickened_dict[str(i+1)])
            vol_fn = thickened_dict[str(i+1)]
            print('\t\tSlab =',i+1)
            print('\t\tSurf fn:',surf_fn['upsample_h5'])
            print('\t\tVol fn:',vol_fn)
            
            # read surface coordinate values from file
            coords_h5 = h5py.File(surf_fn['upsample_h5'],'r')
            coords = coords_h5['data'][:]

            # read slab volume
            img = nib.load(vol_fn)
            vol = img.get_fdata()

            assert np.max(vol) != np.min(vol) , f'Error: empty volume {vol_fn}; {np.max(vol) != np.min(vol)} '

            # set variable names for coordinate system
            xstart = img.affine[0,3]
            ystart = img.affine[1,3]
            zstart = img.affine[2,3]
            xstep = np.abs(img.affine[0,0])
            ystep = np.abs(img.affine[1,1])
            zstep = np.abs(img.affine[2,2])

            # get the intervals along the y-axis of the volume where
            # we have voxel intensities that should be interpolated onto the surfaces
            intervals_voxel = get_section_intervals(vol)

            #convert from voxel values to real world coordinates
            
            for iv in intervals_voxel : 
                #plt.figure(figsize=(9,9))
                #plt.imshow(vol[:,iv[0],:])

                for y0 in range(iv[0],iv[1]+1):
                    y1 = y0 + 1
                    y0w = y0 * ystep + ystart 
                    y1w = y1 * ystep + ystart 
                    
                    #the voxel values should be the same along the y-axis within an interval
                    #WARNING: this will NOT work if there isn't a gap between thickened autoradiograph sections!
                    section = vol[:,y0,:]
                    assert np.max(section) != np.min(section), 'Error: emptry section before interpolation '
                    valid_coords_world, valid_coords_idx = get_valid_coords( coords, [y0w,y1w])

                    if valid_coords_world.shape[0] != 0  :
                        x = np.rint( (valid_coords_world[:,0] - xstart)/xstep ).astype(int)
                        z = np.rint( (valid_coords_world[:,2] - zstart)/zstep ).astype(int)

                        xmax = np.max(x)
                        zmax = np.max(z)

                        #plt.scatter(z,x,s=1,c='r',marker='.')


                        assert zmax < section.shape[1] , f'Error: z index {zmax} is greater than dimension {section.shape[1]}'
                        assert xmax < section.shape[0] , f'Error: x index {xmax} is greater than dimension {section.shape[0]}'
                        # get nearest neighbour voxel intensities at x and z coordinate locations
                        values = section[x,z]


                        #plt.savefig(f'{qc_dir}/{iv[0]}_{iv[1]}.png', dpi=400)

                        assert np.sum(np.isnan(values)) == 0 , f'Error: nan found in values from {vol_fn}'

                        assert np.mean(values) > np.min(vol), f'Error: empty section {y0} in {vol_fn}'
                        all_values[valid_coords_idx] = values 

                #plt.savefig(f'{qc_dir}/{iv[0]}_{iv[1]}.png', dpi=400)
                #print('\tSaved', f'{qc_dir}/{iv[0]}_{iv[1]}.png')
                #plt.clf()
                #plt.cla()

            #DEBUG
            #threshold=1000
            #all_values[all_values < threshold] = 0
            np.savetxt(interp_csv, all_values)


def thicken_sections(interp_dir, slab_dict, df_ligand, resolution, tissue_type='' ):
    rec_thickened_dict = {} 
    target_file=f'nl_2d_vol{tissue_type}_fn'
    slab_dict_keys = slab_dict.keys() 
    slab_dict_keys = sorted(slab_dict_keys)
    
    for  i in slab_dict_keys :
        slab = slab_dict[i]

        slab_df = df_ligand.loc[df_ligand['slab'].astype(int)==int(i)]
        source_image_fn = slab[target_file]
        print('Source image for thickening:', source_image_fn)
        ligand = slab_df['ligand'].values[0]
        thickened_fn = f'{interp_dir}thickened_{int(i)}_{ligand}_{resolution}{tissue_type}.nii.gz'

        print('\t\tThickened Filename', thickened_fn)
        
        if not os.path.exists(thickened_fn) :
            array_img = nib.load(source_image_fn)
            array_src = array_img.get_fdata()
            
            assert np.sum(array_src) != 0, 'Error: source volume for thickening sections is empty\n'+ source_image_fn

            width = np.round(1+1*resolution/(0.02*2)).astype(int)

            print('\t\tThickening sections to ',0.02*width*2)
            dim=[array_src.shape[0], 1, array_src.shape[2]]
            rec_vol = np.zeros_like(array_src)
            for row_i, row in slab_df.iterrows() : 
                y = int(row['volume_order'])
                
                # Conversion of radioactivity values to receptor density values
                section = array_src[:, y, :].copy()

                if np.sum(section) == 0 : 
                    print(f'Warning: empty frame {i} {row["crop_fn"]}\n')
                np.sum(section) 
                section = section.reshape(dim)
                if row['conversion_factor'] > 0 and tissue_type != '_cls' :
                     section *= row['conversion_factor']
                elif row['conversion_factor'] == 0 : 
                    continue
                


                y0 = int(y)-width if int(y)-width > 0 else 0
                y1 = 1+int(y)+width if int(y)+width <= array_src.shape[1] else array_src.shape[1]
                #put ligand sections into rec_vol
                print(y0,y1)
                rec_vol[:, y0:y1, :] = np.repeat(section, y1-y0, axis=1)

            assert np.sum(rec_vol) != 0, 'Error: receptor volume for single ligand is empty\n'
            nib.Nifti1Image(rec_vol, array_img.affine).to_filename(thickened_fn)

        print('\t-->',i,thickened_fn)
        rec_thickened_dict[i] = thickened_fn

    return rec_thickened_dict


def create_thickened_volumes(interp_dir, depth_list, depth_fn_slab_space, depth_fn_list, slab_dict, df_ligand, resolution, tissue_type=''):

    for depth_index, (depth, depth_fn) in enumerate(zip(depth_list,depth_fn_list)):
        # Get surfaces transformed into slab space
        slabs = list(slab_dict.keys())
        slabs.sort()
        surf_fn_list = [ depth_fn_slab_space[i][depth] for i in slabs  ]
        
        # 1. Thicken sections
        thickened_dict = thicken_sections(interp_dir, slab_dict, df_ligand, resolution, tissue_type=tissue_type)

        
        thickened_dict = combat_slab_normalization(df_ligand, thickened_dict)

        # 2. Project autoradiograph densities onto surfaces
        print('\t\t\t\tProjecting volume to surface.')
        project_volumes_to_surfaces(surf_fn_list, thickened_dict, depth_fn, interp_dir)
    return thickened_dict

def get_profiles(profiles_fn, recon_out_prefix, depth_fn_mni_space, depth_list, depth_fn_list, nrows):
    print('\tGetting profiles')

    # 3. Interpolate missing densities over surface
    if not os.path.exists(profiles_fn) :
        
        profiles = h5py.File(profiles_fn, 'w') 
        
        profiles.create_dataset('data', (nrows, len(depth_list)) )

        for depth_index, (depth, depth_fn) in enumerate(zip(depth_list, depth_fn_list)):
            print('\t\t\t\treading interpolated values from ',depth_fn)
            profiles_raw = pd.read_csv(depth_fn, header=None, index_col=None)

            sphere_rsl_fn = depth_fn_mni_space[depth]['sphere_rsl_fn'] 
            surface_val = profiles_raw.values.reshape(-1,) 
          
            profile_vector = interpolate_over_surface(sphere_rsl_fn, surface_val,threshold=0.02)

            profiles['data'][:,depth_index] = profile_vector
            del profile_vector

    return profiles_fn

def project_volume_to_surfaces(interp_dir, surf_dir, depth_list, slab_dict, df_ligand, depth_fn_mni_space, depth_fn_slab_space, resolution, recon_out_prefix, tissue_type='', clobber=False):
    
    profiles_fn = f'{recon_out_prefix}_profiles.h5'
    
    example_depth_fn = depth_fn_mni_space[depth_list[0]]['upsample_fn']
    nrows = h5py.File(example_depth_fn)['data'].shape[0]

    depth_fn_list = [ sub('.h5', f'_{depth}_raw.csv', profiles_fn) for depth in depth_list ]
   
    thickened_dict = create_thickened_volumes(interp_dir, depth_list, depth_fn_slab_space, depth_fn_list, slab_dict, df_ligand, resolution, tissue_type=tissue_type)
    
    profiles_fn = get_profiles(profiles_fn, recon_out_prefix, depth_fn_mni_space, depth_list, depth_fn_list, nrows)

    return thickened_dict, profiles_fn
     
def interpolate_over_surface(sphere_obj_fn,surface_val,threshold=0):
    print('\t\tInterpolating Over Surface')
    print('\t\t\tSphere fn:',sphere_obj_fn)
    # get coordinates from dicitonary with mesh info
    coords = h5py.File(sphere_obj_fn)['data'][:] 

    spherical_coords = surface_tools.spherical_np(coords) 

    #define a mask of verticies where we have receptor densitiies
    surface_mask = surface_val > threshold * np.max(surface_val)
    assert np.sum(surface_mask) != 0, "Error, empty profiles {}".format(np.sum(surface_mask))
    #define vector with receptor densities 
    surface_val_src = surface_val[ surface_mask.astype(bool) ]

    #define vector without receptor densities
    surface_val_tgt = surface_val[ ~surface_mask.astype(bool) ]

    # get coordinates for vertices in mask
    spherical_coords_src = spherical_coords[ surface_mask.astype(bool), : ]
    
    # get spherical coordinates from cortical mesh vertex coordinates
    lats_src, lons_src = spherical_coords_src[:,1]-np.pi/2, spherical_coords_src[:,2]

    temp = np.concatenate([(spherical_coords_src[:,1]-np.pi/2).reshape(-1,1), spherical_coords_src[:,2].reshape(-1,1)],axis=1)

    # create mesh data structure

    temp = np.concatenate([lons_src.reshape(-1,1),lats_src.reshape(-1,1)],axis=1)

    mesh = stripy.sTriangulation(lons_src, lats_src)
    lats, lons = spherical_coords[:,1]-np.pi/2, spherical_coords[:,2]
    
    # interpolate over the sphere
    #if False :
    interp_val, interp_type = mesh.interpolate(lons,lats, zdata=surface_val[surface_mask], order=1)
    #else :
    #    interp_val = krig(lons, lats, lats_src, lats_src, values )

    # interpolate over the sphere
        
    return interp_val

def transform_surf_to_slab(interp_dir, slab_dict, depth_fn_space_mni, ref_gii_fn,  clobber=0):
    surf_rsl_dict={}
    for slab, cur_slab_dict in slab_dict.items() :
        surf_rsl_dict[slab]={}
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
                apply_ants_transform_to_gii(upsample_fn, [cur_slab_dict['nl_3d_tfm_fn']], upsample_slab_space_fn, [0], depth_fn_space_mni[0]['faces_fn'], ref_gii_fn)
     
    return surf_rsl_dict

def prepare_surfaces(slab_dict, depth_list, interp_dir, resolution, surf_dir='civet/mri1/surfaces/surfaces/', n_vertices = 327696, clobber=0):
    '''

    '''
    surf_rsl_dir = interp_dir +'/surfaces/' 
    os.makedirs(surf_rsl_dir, exist_ok=True)
    
    #Interpolate at coordinate locations
    ref_surf_fn = surf_base_str.format(surf_dir,'gray', n_vertices,'','surf.gii')
    ref_surf_obj_fn = surf_base_str.format(surf_dir,'gray', n_vertices,'','obj')

    #if not os.path.exists(ref_surf_fn) :
    #    shell('ConvertSurface -i_obj {ref_surf_obj_fn}  -o_gii {ref_surf_fn}')

    surf_gm_obj_fn = surf_base_str.format(surf_dir,'gray', n_vertices,'','obj')
    surf_wm_obj_fn = surf_base_str.format(surf_dir,'white', n_vertices,'','obj')

    surf_gm_fn = surf_base_str.format(surf_rsl_dir,'gray', n_vertices,'','surf.gii')
    surf_wm_fn = surf_base_str.format(surf_rsl_dir,'white', n_vertices,'','surf.gii')
    
    sphere_obj_fn = surf_base_str.format(surf_dir,'mid', n_vertices,'_sphere','surf.gii')
    obj_to_gii(surf_gm_obj_fn, ref_surf_fn, surf_gm_fn)
    obj_to_gii(surf_wm_obj_fn, ref_surf_fn, surf_wm_fn)

    #upsample transformed surfaces to given resolution
    print("\tUpsampling and inflating surfaces.") 
    depth_fn_mni_space = upsample_and_inflate_surfaces(surf_rsl_dir, surf_wm_fn, surf_gm_fn, resolution, depth_list)

    #For each slab, transform the mesh surface to the receptor space
    #TODO: transform points into space of individual autoradiographs
    print("\tTransforming surfaces to slab space.") 
    depth_fn_slab_space = transform_surf_to_slab(surf_rsl_dir, slab_dict, depth_fn_mni_space, surf_wm_fn)

    return depth_fn_mni_space, depth_fn_slab_space

class ImageParameters():
    def __init__(self, starts, steps, dimensions):
        self.starts = starts
        self.steps = steps 
        self.dimensions = dimensions
        self.affine = np.array([[self.steps[0], 0, 0, self.starts[0]],
                       [0, self.steps[1], 0, self.starts[1]],
                       [0, 0, self.steps[2], self.starts[2]],
                       [0, 0, 0, 1]])
        print('\tStarts', self.starts)
        print('\tSteps', self.steps)
        print('\tDimensions:', self.dimensions)
        print('\tAffine', self.affine)


def get_image_parameters(fn, resolution):
    img = nib.load(fn)

    starts = np.array(img.affine[[0,1,2],3])

    steps_lo = [resolution]*3
    steps_hi = np.abs(img.affine[[0,1,2],[0,1,2]])

    dimensions_hi = img.shape
    dimensions_lo = np.array([ img.shape[0] * img.affine[0,0]/resolution, 
                            img.shape[1] * img.affine[1,1]/resolution, 
                            img.shape[2] * img.affine[2,2]/resolution]).astype(int)


    imageParamLo = ImageParameters(starts, steps_lo, dimensions_lo)
    imageParamHi = ImageParameters(starts, steps_hi, dimensions_hi)

    return imageParamLo, imageParamHi

def combine_interpolated_sections(slab_fn, vol_interp, df_ligand_slab, ystart, ystep, ystep_hires=0.02) :
    img = nib.load(slab_fn)
    vol = img.get_fdata()

    volume_order_list = df_ligand_slab['volume_order'].values.astype(int)
    ymin = min(volume_order_list) 
    ymax = max(volume_order_list) 
    for y in range(vol.shape[1]) :
        yw = y * np.abs(ystep_hires) + ystart
        yi = np.rint( (yw - ystart) / ystep ).astype(int)
        if np.sum(vol[:,y,:]) <= 0 and y > ymin and y < ymax:
            interp_section = vol_interp[ : , yi, : ]
            vol[:, y, :] = interp_section
            print('Adding interpolated_section:',y, np.rint(np.sum(interp_section)))
        #else :
        #    print('Not including section', np.sum(vol[:,y,:]))

    return vol


def create_reconstructed_volume(interp_fn_list, interp_dir, thickened_fn_dict, profiles_fn, depth_list, depth_fn_slab_space, args, files, resolution, df_ligand, use_mapper=True, clobber=False):
    profiles = None

    for interp_fn, slab in zip(interp_fn_list, args.slabs) :

        if not os.path.exists(interp_fn) or clobber : 
            
            if type(profiles) != type(np.array) : profiles = h5py.File(profiles_fn, 'r')['data'][:]

            # Load dimensions for output volume
            files_resolution = files[str(int(slab))]
            resolution_list = list(files_resolution.keys())
            max_resolution = resolution_list[-1]
            slab_fn = files_resolution[ max_resolution ]['nl_2d_vol_fn'] 
            thickened_fn = thickened_fn_dict[ slab ]

            imageParamLo, imageParamHi = get_image_parameters(slab_fn, float(max_resolution))
            
            if use_mapper :
                #Create an object that will be used to interpolate over the surfaces
                wm_upsample_fn = depth_fn_slab_space[slab][depth_list[0]]['upsample_gii']
                gm_upsample_fn = depth_fn_slab_space[slab][depth_list[-1]]['upsample_gii']
                print('\twm', wm_upsample_fn)
                print('\tgm', gm_upsample_fn)
                npz_dir = interp_dir+f'/npz-{slab}_{resolution}mm/'
                os.makedirs(npz_dir, exist_ok=True)
                mapper = SurfaceVolumeMapper(white_surf=wm_upsample_fn, gray_surf=gm_upsample_fn, 
                            resolution=imageParamLo.steps, dimensions=imageParamLo.dimensions, 
                            origin=imageParamLo.starts, filename=None, save_in_absence=False, mask=None,
                            out_dir=npz_dir, left_oriented=False )

                vol_interp = mapper.map_profiles_to_block(profiles, interpolation='linear')

            else :
                vol_interp = multi_mesh_to_volume(profiles, depth_fn_slab_space[slab], depth_list, imageParamHi.dimensions, imageParamHi.starts, imageParamHi.steps)
            assert np.sum(vol_interp) > 0 , f'Error: interpolated volume is empty using {profiles_fn}'

            df_ligand_slab = df_ligand.loc[ df_ligand['slab'].astype(int)==int(slab)]
            output_vol = combine_interpolated_sections(thickened_fn, vol_interp, df_ligand_slab, imageParamLo.starts[1], imageParamLo.steps[1])

            print(f'\tWrite volumetric interpolated values to {interp_fn} ',end='\t')
            #nib.Nifti1Image(vol_interp, imageParamLo.affine).to_filename(interp_fn)
            # prefilter the output volume for when it gets resampled to mni space
            output_vol = gaussian_filter(output_vol, [0, (resolution/0.02)/ np.pi, 0 ] )
            nib.Nifti1Image(output_vol, imageParamHi.affine ).to_filename(interp_fn)
            print('\nDone.')
        else :
            print(interp_fn, 'already exists')

def interpolate_between_slabs(depth_fn_mni_space, depth_list, profiles_fn, interp_dir, srv_rsl_fn, resolution):
    img = nib.load(srv_rsl_fn)
    starts = np.array(img.affine[[0,1,2],3])
    steps = [resolution]*3
    dimensions = np.array([ img.shape[0] * img.affine[0,0]/resolution, 
                            img.shape[1] * img.affine[1,1]/resolution, 
                            img.shape[2] * img.affine[2,2]/resolution]).astype(int)

    wm_upsample_fn = depth_fn_mni_space[depth_list[0]]['upsample_gii_fn']
    gm_upsample_fn = depth_fn_mni_space[depth_list[-1]]['upsample_gii_fn']

    npz_dir = interp_dir+f'/npz-mni/'

    os.makedirs(npz_dir, exist_ok=True)
    
    mapper = SurfaceVolumeMapper(white_surf=wm_upsample_fn, gray_surf=gm_upsample_fn, 
                resolution=steps, dimensions=dimensions, 
                origin=starts, filename=None, save_in_absence=False, mask=None,
                out_dir=npz_dir, left_oriented=False )

    profiles = h5py.File(profiles_fn, 'r')['data'][:]
    vol_interp = mapper.map_profiles_to_block(profiles, interpolation='linear')

    return vol_interp

def combine_slabs_to_volume(interp_fn_mni_list, output_fn):
    ref_img = nib.load(interp_fn_mni_list[0])
    output_volume = np.zeros(ref_img.shape)

    if not os.path.exists(output_fn) :
        for interp_mni_fn in interp_fn_mni_list :
            vol = nib.load(interp_mni_fn).get_fdata()
            output_volume += vol
        nib.Nifti1Image(output_volume, ref_img.affine ).to_filename( output_fn )

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

def transform_slab_to_mni(slabs, slab_dict, mni_fn,  template_out_prefix):

    for slab in slabs :
        slab_out_prefix = sub('_slab_', f'_{slab}_', template_out_prefix)
        interp_fn  = f'{slab_out_prefix}_space-slab.nii.gz'
        interp_space_mni_fn  = f'{slab_out_prefix}_space-mni.nii.gz'
        tfm = slab_dict[slab]['nl_3d_tfm_fn']
        if not os.path.exists(interp_space_mni_fn) :
            shell(f'antsApplyTransforms -d 3 -n NearestNeighbor -i {interp_fn} -r {mni_fn} -t {tfm} -o {interp_space_mni_fn}')

def create_final_reconstructed_volume(final_mni_fn, mni_fn, resolution,  depth_fn_mni_space, depth_list, interp_dir, interp_fn_mni_list, recon_out_prefix, profiles_fn ):
    combined_slab_mni_fn = f'{recon_out_prefix}_space-mni_not_filled.nii.gz'
    surf_interp_mni_fn = f'{recon_out_prefix}_surf-interp_space-mni.nii.gz'

    ref_img = nib.load(mni_fn) 
    ystart = ref_img.affine[1,3]
    combine_slabs_to_volume(interp_fn_mni_list, combined_slab_mni_fn)

    print('\tInterpolate between slabs')
    vol_interp = interpolate_between_slabs(depth_fn_mni_space, depth_list, profiles_fn, interp_dir, mni_fn, resolution)

    print('\tWriting surface interpolation volume:\n\t\t', final_mni_fn)
    nib.Nifti1Image(vol_interp, ref_img.affine).to_filename(surf_interp_mni_fn)
    
    # fill in missing sections in whole brain
    print('\tCombine interpoalted section with slabs to fill gaps')
    combined_slab_vol = nib.load(combined_slab_mni_fn).get_fdata()

    idx = combined_slab_vol <= 300
    combined_slab_vol[ idx ] = vol_interp[ idx ]
    output_vol = combined_slab_vol

    # save final volume 
    print('\tWriting', final_mni_fn)
    nib.Nifti1Image(output_vol, ref_img.affine).to_filename(final_mni_fn)


def surface_interpolation(df_ligand, slab_dict, out_dir, interp_dir, brain, hemi, resolution, mni_fn, args, files,  n_depths=3, tissue_type='', surf_dir='civet/mri1/surfaces/surfaces/', n_vertices = 327696, clobber=0):

    ligand = df_ligand['ligand'].values[0]

    template_out_prefix=f'{interp_dir}/{brain}_{hemi}_slab_{ligand}_{resolution}mm_l{n_depths}{tissue_type}'
    recon_out_prefix=f'{interp_dir}/{brain}_{hemi}_{ligand}_{resolution}mm{tissue_type}'
    #make sure resolution is interpreted as float
    resolution=float(resolution) 
    
    os.makedirs(interp_dir, exist_ok=True)

    #set depths
    dt = 1.0/ n_depths
    depth_list = np.arange(0, 1+dt/10, dt)
    depth_list = np.insert(depth_list,0, 0)
    print('\tPreparing surfaces for surface-interpolation')
    depth_fn_mni_space, depth_fn_slab_space = prepare_surfaces(slab_dict, depth_list, interp_dir, resolution, surf_dir=surf_dir, n_vertices = n_vertices, clobber=clobber)


    print('\tReconstructing', ligand)
    interp_fn_list, interp_fn_mni_list = find_ligands_to_reconstruction(args.slabs, interp_dir, template_out_prefix)
    
    print('\tInterpolating for ligand:',ligand)

    # Extract profiles from the slabs using the surfaces 
    thickened_dict, profiles_fn = project_volume_to_surfaces(interp_dir, interp_dir+'/surfaces/', depth_list, slab_dict, df_ligand, depth_fn_mni_space, depth_fn_slab_space, resolution,  recon_out_prefix, tissue_type=tissue_type)
    
    # Interpolate a 3D receptor volume from the surface mesh profiles
    print('\tCreate Reconstructed Volume')
    create_reconstructed_volume(interp_fn_list, interp_dir, thickened_dict, profiles_fn, depth_list, depth_fn_slab_space, args, files, resolution, df_ligand, use_mapper=True, clobber=clobber)

    # transform interp_fn to mni space
    print('\tTransform slab to mni')
    transform_slab_to_mni(args.slabs, slab_dict,mni_fn, template_out_prefix)
    
    final_mni_fn = f'{recon_out_prefix}_space-mni.nii.gz'
    # interpolate from surface to volume over entire brain
    print('\tCreate final reconstructed volume')
    if not os.path.exists(final_mni_fn):
        create_final_reconstructed_volume(final_mni_fn, mni_fn, resolution, depth_fn_mni_space, depth_list, interp_dir, interp_fn_mni_list, recon_out_prefix, profiles_fn )


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
