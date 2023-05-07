import bisect
import contextlib
import re
import scipy
import os
import pandas as pd
import imageio
import utils.ants_nibabel as nib
import nibabel as nb
import PIL
import matplotlib
import time
import ants
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import h5py as h5
import multiprocessing
import nibabel
import debug
from glob import glob
from re import sub
from joblib import Parallel, delayed
from utils.utils import shell, w2v, v2w, get_section_intervals, prefilter_and_downsample
from utils.mesh_io import save_mesh, load_mesh, save_obj, read_obj
from utils.fit_transform_to_paired_points import fit_transform_to_paired_points
from ants import get_center_of_mass
from nibabel.processing import resample_from_to
from scipy.ndimage.filters import gaussian_filter 
from os.path import basename
from subprocess import call, Popen, PIPE, STDOUT
from sklearn.cluster import KMeans
from scipy.ndimage import zoom
from skimage.transform import resize
from scipy.ndimage import label, center_of_mass
from time import time
from utils.reconstruction_classes import SlabReconstructionData

os_info = os.uname()
global num_cores
if os_info[1] == 'imenb079':
    num_cores = 1 
else :
    num_cores = min(14, multiprocessing.cpu_count() )

def get_surf_from_dict(d):
    keys = d.keys()
    if 'upsample_h5' in keys : 
        surf_fn = d['upsample_h5']
    elif 'depth_rsl_fn' in keys :
        surf_fn = d['depth_rsl_fn']
    elif 'surf' in keys :
        surf_fn = d['surf']
    else : 
        assert False, f'Error: could not find surface in keys, {keys}'
    return surf_fn



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
    
    x = np.rint( (coords[:,0] - starts[0]) / steps[0] ).astype(int)
    y = np.rint( (coords[:,1] - starts[1]) / steps[1] ).astype(int)
    z = np.rint( (coords[:,2] - starts[2]) / steps[2] ).astype(int)

    idx = (x >= 0) & (y >= 0) & (z >= 0) & (x < dimensions[0]) & ( y < dimensions[1]) & ( z < dimensions[2] )

    #perc_mesh_in_volume = np.sum(~idx)/idx.shape[0]
    #assert perc_mesh_in_volume < 0.1, f'Error: significant portion ({perc_mesh_in_volume}) of mesh outside of volume '
    
    assert np.sum(idx) > 0, 'Assert: no voxels found inside mesh_to_volume'
    x = x[idx]
    y = y[idx]
    z = z[idx]

    vertex_values = vertex_values[idx] 

    for i, (xc, yc, zc) in enumerate(zip(x,y,z)) :

        if debug.DEBUG == 4 :
            interp_vol[xc,yc,zc] += yc
        else :
            interp_vol[xc,yc,zc] += vertex_values[i]
    
        n_vol[xc,yc,zc] += 1

    return interp_vol, n_vol



def multi_mesh_to_volume(profiles, surfaces, depth_list, dimensions, starts, steps, resolution, y0, y1, origin=[0,0,0], ref_fn=None):
    all_points=[]
    all_values=[]
    interp_vol = np.zeros(dimensions)
    n_vol = np.zeros_like(interp_vol)

    slab_start = min(y0,y1)
    slab_end = max(y0,y1)

    for ii in range(profiles.shape[1]) :
        print(ii)
        print(depth_list)
        print(surfaces.keys())
        surf_fn = surfaces[depth_list[ii]]#DEBUG['depth_rsl_fn']
        print('\tSURF', surf_fn);

        if 'npz' in os.path.splitext(surf_fn)[-1] : ext = '.npz'
        else : ext='.surf.gii'
        #out_fn = re.sub(ext, f'_upsampled-face-{ii}', surf_fn)
        #to_do_list.append((surf_fn,out_fn))
        #to_do_ist.append((surf_fn,surf_fn))
        points = np.load(surf_fn)['points']
        assert points.shape[0] == profiles.shape[0], 'Error mismatch in number of points between {surf_fn} and vertex values file'
        print(np.sum(np.abs(profiles[:,ii])))

        interp_vol, n_vol = mesh_to_volume(points, profiles[:,ii], dimensions, starts, steps, interp_vol=interp_vol, n_vol=n_vol)
    
    interp_vol[ n_vol>0 ] = interp_vol[n_vol>0] / n_vol[n_vol>0]
    
    assert np.sum(np.abs(interp_vol)) != 0 , 'Error: interpolated volume is empty'
    return interp_vol




def add_entry(d,i, lst):
    try :
        d[i] += lst
    except KeyError:
        d[i]=list(lst)
    return d[i]

def unique_points(points, scale=1000000000):
    #rpoints = np.rint(points * scale).astype(np.int64)
    upoints, unique_index, unique_inverse = np.unique(points.astype(np.float128).round(decimals=3),axis=0,return_index=True, return_inverse=True)

    return points[unique_index,:], unique_index, unique_inverse

def upsample_over_faces(surf_fn, resolution, out_fn,  face_mask=None, profiles_vtr=None, slab_start=None, slab_end=None, ref_faces=None) :
    print(surf_fn)
    coords, faces = load_mesh_ext(surf_fn)

    if type(faces) == type(None):
        if type(ref_faces) != type(None) :
            del faces
            faces=ref_faces
        else :
            print('Error: ref faces not defined')

    if type(face_mask) != np.ndarray :
        face_mask=np.ones(faces.shape[0]).astype(np.bool)

    write_surface=False 
    if '.surf.gii' in out_fn: write_surface=True

    #Choice 1: truncate vertices by volume boundaries OR by valid y sections where histological
    #sections have been acquired
    if type(face_mask) == None :
        if slab_start == None : slab_start = min(coords[:,1])
        if slab_end == None : slab_end = max(coords[:,1])
        #find the vertices that are inside of the slab
        valid_idx = np.where( (coords[:,1] >= slab_start) & (coords[:,1] <= slab_end) )[0]
        #create a temporary array for the coords where the exluded vertices are equal NaN
        # this is necessary because we only want to upsample a subset of the entire mesh
        new_coords = np.zeros_like(coords)
        new_coords[:] = np.NaN
        new_coords[valid_idx,:] = coords[valid_idx]
        face_coords = face_coords[valid_faces_idx ,:]
        face_mask = np.where( ~ np.isnan(np.sum(face_coords,axis=(1,2))) )[0]
    else : 
        target_faces = faces[face_mask]
        face_coords = coords[faces]
    #del coords

    # Choice 2 : if values are provided, interpolate these over the face, otherwise create 0-array
    if type(profiles_vtr) != type(None) :
        face_vertex_values = profiles_vtr[faces] 
        face_vertex_values = face_vertex_values[face_mask,:]
    else :
        face_vertex_values = np.zeros([face_coords.shape[0],3])

    points, values, new_points_gen = calculate_upsampled_points(faces, face_coords, face_vertex_values, resolution)

    assert points.shape[1]==3, 'Error: shape of points is incorrect ' + points.shape 
    points, unique_index, unique_reverse = unique_points(points)

    np.savez(out_fn, points=points, values=values)
     
    new_points_gen = [ new_points_gen[i] for i in unique_index ]

    #for i in range(points.shape[0]):
    #    new_points_gen[i].idx = i
    #    print(points[i])
    #    print( new_points_gen[i].generate_point(coords) )
    #    print()

    print('\t\tSaved', out_fn)
    assert len(new_points_gen) == points.shape[0], f'Error: the amount of points does not equal the amount of point generators {len(new_points_gen)} vs {points.shape[0]}'

    return points, values, new_points_gen


def get_faces_from_neighbours(ngh):
    face_dict={}
    print('\tCreate Faces')
    for i in range(len(ngh.keys())):
        if i % 1000 : print(f'2. {100*i/ngh.shape[0]} %', end='\r')
        for ngh0 in ngh[i] :
            for ngh1 in ngh[ngh0] :
                print(i, ngh0, ngh1)
                if ngh1 in ngh[i] :
                    face = [i,ngh0,ngh1]
                    face.sort()
                    face_str = sorted_str(face)
                    try :
                        face_dict[face_str]
                    except KeyError:
                        face_dict[face_str] = face

    n_faces = len(face_dict.keys())

    faces = np.zeros(n_faces,3)
    for i, f in enumerate(faces.values()) : faces[i] = f

    return faces 

def get_triangle_vectors(points):

    v0 = points[1,:] - points[0,:]
    v1 = points[2,:] - points[0,:]
    return v0, v1


def volume_to_surface(coords, volume_fn, values_fn=''):
    print(volume_fn)
    img = nibabel.load(volume_fn)
    vol = img.get_fdata()
    
    starts = img.affine[[0,1,2],3]
    step = np.abs(img.affine[[0,1,2],[0,1,2]]) 
    dimensions = vol.shape
    
    interp_vol, _  = mesh_to_volume(coords, np.ones(coords.shape[0]), dimensions, starts, img.affine[[0,1,2],[0,1,2]])
    nib.Nifti1Image(interp_vol.astype(np.float32),nib.load(volume_fn).affine).to_filename('tmp.nii.gz')
    nib.load('tmp.nii.gz')

    coords_idx = np.rint((coords - starts) / step).astype(int)
    
    idx0 = (coords_idx[:,0] >= 0) & (coords_idx[:,0] < dimensions[0])
    idx1 = (coords_idx[:,1] >= 0) & (coords_idx[:,1] < dimensions[1]) 
    idx2 = (coords_idx[:,2] >= 0) & (coords_idx[:,2] < dimensions[2])
    idx_range = np.arange(coords_idx.shape[0]).astype(int)

    idx = idx_range[idx0 & idx1 & idx2]

    coords_idx = coords_idx[ idx0 & idx1 & idx2 ]

    print(np.max(coords_idx[:,0]), np.max(coords_idx[:,1]), np.max(coords_idx[:,2]))
    print(dimensions)

    values = vol[coords_idx[:,0],coords_idx[:,1],coords_idx[:,2]]

    if values_fn != '' :
        pd.DataFrame(values).to_filename(values_fn, index=False, header=False)

    return values, idx


def mult_vector(v0,v1,x,y,p):
    v0 = v0.astype(np.float128)
    v1 = v1.astype(np.float128)
    x = x.astype(np.float128)
    y = y.astype(np.float128)
    p = p.astype(np.float128)

    mult = lambda a,b : np.multiply(np.repeat(a.reshape(a.shape[0],1),b.shape,axis=1), b).T
    w0=mult(v0,x).astype(np.float128)
    w1=mult(v1,y).astype(np.float128)
    # add the two vector components to create points within triangle
    p0 = p + w0 + w1 
    return p0

def interpolate_face(points, values, resolution, output=None, new_points_only=False):
    # calculate vector on triangle face
    v0, v1 = get_triangle_vectors(points.astype(np.float128))

    #calculate the magnitude of the vector and divide by the resolution to get number of 
    #points along edge
    calc_n = lambda v : np.ceil( np.sqrt(np.sum(np.power(v,2)))/resolution).astype(int)
    mag_0 = calc_n(v0)
    mag_1 = calc_n(v1)

    n0 = max(2,mag_0) #want at least the start and end points of the edge between two vertices
    n1 = max(2,mag_1)

    #calculate the spacing from 0 to 100% of the edge
    l0 = np.linspace(0,1,n0).astype(np.float128)
    l1 = np.linspace(0,1,n1).astype(np.float128)
    
    #create a percentage grid for the edges
    xx, yy = np.meshgrid(l1,l0)
    
    #create flattened grids for x, y , and z coordinates
    x = xx.ravel()
    y = yy.ravel()
    z = 1- np.add(x,y)

    valid_idx = x+y<=1.0 #eliminate points that are outside the triangle
    x = x[valid_idx] 
    y = y[valid_idx]
    z = z[valid_idx]

    # multiply edge by the percentage grid so that we scale the vector
    p0 = mult_vector(v0,v1,x,y,points[0,:].astype(np.float128))
    
    interp_values = values[0]*x + values[1]*y + values[2]*z
    '''
    if new_points_only : 
        filter_arr = np.ones(p0.shape[0]).astype(bool)
        dif = lambda x,y : np.abs(x-y)<0.0001
        ex0= np.where( (dif(p0,points[0])).all(axis=1) )[0][0]
        ex1= np.where( (dif(p0,points[1])).all(axis=1) )[0][0]
        ex2 = np.where((dif(p0,points[2])).all(axis=1) )[0][0]
        filter_arr[ ex0 ] = filter_arr[ex1] = filter_arr[ex2] = False

        p0 = p0[filter_arr]
        interp_values = interp_values[filter_arr]
    '''
    return p0, interp_values, x, y 

class NewPointGenerator():
    def __init__(self, idx, face, x, y):
        self.idx = idx
        self.face = face
        self.x = x.astype(np.float128)
        self.y = y.astype(np.float128)
    
    def generate_point(self, points) :
        cur_points = points[self.face].astype(np.float128)

        v0, v1 = get_triangle_vectors(cur_points)
    
        #new_point = mult_vector(v0, v1, self.x, self.y, points[0,:])
        comp0 = v0.astype(np.float128) * self.x.astype(np.float128)
        comp1 = v1.astype(np.float128) * self.y.astype(np.float128)
        #print('vector components', comp0, comp1, cur_points[0,:])
        new_point = comp0 + comp1 + cur_points[0,:] 

        return new_point
        

def calculate_upsampled_points(faces,  face_coords, face_vertex_values, resolution, new_points_only=False):
    points=np.zeros([face_coords.shape[0]*5,3],dtype=np.float128)
    values=np.zeros([face_coords.shape[0]*5])
    n_points=0
    new_points_gen = {}

    for f in range(face_coords.shape[0]):
        if f % 1000 == 0 : print(f'\t\tUpsampling Faces: {100.*f/face_coords.shape[0]:.3}',end='\r')
        
        p0, v0, x, y = interpolate_face(face_coords[f], face_vertex_values[f], resolution*0.9, new_points_only=new_points_only)
        
        if n_points + p0.shape[0] >= points.shape[0]:
            points = np.concatenate([points,np.zeros([face_coords.shape[0],3]).astype(np.float128)], axis=0)
            values = np.concatenate([values,np.zeros(face_coords.shape[0])], axis=0)
       
        new_indices = n_points + np.arange(p0.shape[0]).astype(int)
        cur_faces = faces[f]


        for i, idx in enumerate(new_indices) : 
            new_points_gen[idx] = NewPointGenerator(idx,cur_faces,x[i],y[i])

        points[n_points:(n_points+p0.shape[0])] = p0
        values[n_points:(n_points+v0.shape[0])] = v0
        n_points += p0.shape[0]
   
    points=points[0:n_points]
    values=values[0:n_points]


    return points, values, new_points_gen


def transform_surface_to_slabs( volume_dict, depth, slab_dict, out_dir, surf_fn, ref_gii_fn=None, faces_fn=None, ext='.surf.gii'):

    for slab, curr_dict in slab_dict.items() :
        thickened_fn = volume_dict[int(slab)]
        nl_3d_tfm_fn = slab_dict[int(slab)]['nl_3d_tfm_fn']
        surf_slab_space_fn = f'{out_dir}/slab-{slab}_{os.path.basename(surf_fn)}' 
        
        #slabData.surfaces[slab][depth] = surf_slab_space_fn
        #slabData.volumes[slab][depth] = thickened_fn

        print('\tFROM:', surf_fn)
        print('\tTO:', surf_slab_space_fn)
        print('\tWITH:', nl_3d_tfm_fn)

        if not os.path.exists(surf_slab_space_fn) :
            apply_ants_transform_to_gii(surf_fn, [nl_3d_tfm_fn], surf_slab_space_fn, 0, faces_fn=faces_fn, ref_gii_fn=ref_gii_fn, ref_vol_fn=thickened_fn)



def load_mesh_ext(in_fn, faces_fn='', correct_offset=False):
    #TODO: move to mesh_io.py
    ext = os.path.splitext(in_fn)[1]
    faces=None
    volume_info = None

    if ext in ['.pial', '.white', '.gii', '.sphere', '.inflated'] : 
        coords, faces, volume_info = load_mesh(in_fn,correct_offset=correct_offset)
    elif  ext == '.npz' :
        coords = np.load(in_fn)['points']
    else :
        coords = h5.File(in_fn)['data'][:]
        if os.path.splitext(faces_fn)[1] == '.h5' :
            faces_h5=h5.File(faces_fn,'r')
            faces = faces_h5['data'][:]
    return coords, faces

def apply_ants_transform_to_gii( in_gii_fn, tfm_list, out_gii_fn, invert, ref_gii_fn=None, faces_fn=None, ref_vol_fn=None):
    print("transforming", in_gii_fn)
    print("to", out_gii_fn)

    origin = [0,0,0]
    if type(ref_gii_fn) == type(None) :
        ref_gii_fn = in_gii_fn

    if os.path.splitext(ref_gii_fn)[1] in ['.pial', '.white'] : 
        _, _, volume_info = load_mesh(ref_gii_fn)
        #origin = volume_info['cras'] 
    else : volume_info = ref_gii_fn
    
    coords, faces = load_mesh_ext(in_gii_fn)
    
    tfm = ants.read_transform(tfm_list[0])
    flip = 1
    #if np.sum(tfm.fixed_parameters) != 0 : 
    #    print( '/MR1/' in os.path.dirname(in_gii_fn))
    #    if '/MR1/' in os.path.dirname(in_gii_fn):
    #        flipx=flipy=-1
    #        flipz=1
    #        flip_label='MR1'
    #    else :
    flipx=flipy=-1
    flipz=1
    flip_label=f'{flipx}{flipy}{flipz}'
    
    in_file = open(in_gii_fn, 'r')
    
    out_path, out_ext = os.path.splitext(out_gii_fn)
    coord_fn = out_path + f'_{flip_label}_ants_reformat.csv'
    temp_out_fn = out_path + f'_{flip_label}_ants_reformat_warped.csv'
    coords = np.concatenate([coords, np.zeros([coords.shape[0],2])], axis=1 )
    #the for loops are here because it makes it easier to trouble shoot to check how the vertices need to be flipped to be correctly transformed by ants
    #for flipx in [-1]: #[1,-1] :
    #    for flipy in [-1]: #[1,-1]:
    #        for flipz in [1]: #[1,-1]:
    coords[:,0] = flipx*(coords[:,0] -origin[0])
    coords[:,1] = flipy*(coords[:,1] +origin[1])
    coords[:,2] = flipz*(coords[:,2] +origin[2])

    df = pd.DataFrame(coords,columns=['x','y','z','t','label'])
    df.to_csv(coord_fn, columns=['x','y','z','t','label'], header=True, index=False)

    shell(f'antsApplyTransformsToPoints -d 3 -i {coord_fn} -t [{tfm_list[0]},{invert}]  -o {temp_out_fn}',verbose=True)
    df = pd.read_csv(temp_out_fn,index_col=False)
    df['x'] = flipx * (df['x'] - origin[0])
    df['y'] = flipy * (df['y'] - origin[1])
    df['z'] = flipz * (df['z'] - origin[2])

    new_coords = df[['x','y','z']].values
    out_basename, out_ext = os.path.splitext(out_gii_fn)
    
    if out_ext == '.h5':
        f_h5 = h5.File(out_gii_fn, 'w')
        f_h5.create_dataset('data', data=new_coords) 
        f_h5.close()
        save_mesh(out_path+'.surf.gii', new_coords, faces, volume_info=volume_info)
    elif out_ext =='.npz' :
        assert new_coords.shape[1]==3, 'Error: shape of points is incorrect ' + new_coords.shape 
        np.savez(out_basename, points=new_coords)
    else :
        print('\tWriting Transformed Surface:',out_gii_fn, faces.shape )
        save_mesh(out_gii_fn, new_coords, faces, volume_info=volume_info)
    
    nii_fn = out_path +  '.nii.gz'
    if ref_vol_fn != None :
        img = nb.load(ref_vol_fn)
        steps=img.affine[[0,1,2],[0,1,2]]
        starts=img.affine[[0,1,2],3]
        dimensions=img.shape
        interp_vol, _  = mesh_to_volume(new_coords, np.ones(new_coords.shape[0]), dimensions, starts, steps)
        print('\tWriting surface to volume file:',nii_fn)
        nib.Nifti1Image(interp_vol, nib.load(ref_vol_fn).affine).to_filename(nii_fn)
    
    #obj_fn = out_path +  '.obj'
    #save_obj(obj_fn,coords, faces)

