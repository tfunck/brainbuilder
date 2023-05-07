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
from glob import glob
from re import sub
from joblib import Parallel, delayed
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

os_info = os.uname()
global num_cores
if os_info[1] == 'imenb079':
    num_cores = 1 
else :
    num_cores = min(14, multiprocessing.cpu_count() )

global python_version
python_version = 'python3.8'

def kill_python_threads():
    pass
    #shell(f'kill `pidof {python_version} `')
  
  
def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def get_surf_from_dict(d):
    keys = d.keys()
    if 'upsample_h5' in keys : 
        surf_fn = d['upsample_h5']
    elif 'upsample_fn' in keys :
        surf_fn = d['upsample_fn']
    elif 'surf' in keys :
        surf_fn = d['surf']
    else : 
        print('Error: could not find surface in keys,', keys)
        exit(1)
    return surf_fn

def get_edges_from_faces(faces):
    #for convenience create vector for each set of faces 
    f_i = faces[:,0]
    f_j = faces[:,1]
    f_k = faces[:,2]
    
    #combine node pairs together to form edges
    f_ij = np.column_stack([f_i,f_j])
    f_jk = np.column_stack([f_j,f_k])
    f_ki = np.column_stack([f_k,f_i])

    #concatenate the edges into one big array
    edges_all = np.concatenate([f_ij,f_jk, f_ki],axis=0).astype(np.uint32)

    #there are a lot of redundant edges that we can remove
    #first sort the edges within rows because the ordering of the nodes doesn't matter
    edges_all_sorted_0 = np.sort(edges_all,axis=1)
    #create a vector to keep track of vertex number
    edges_all_range= np.arange(edges_all.shape[0]).astype(int)
    #
    edges_all_sorted = np.column_stack([edges_all_sorted_0, edges_all_range ])
    
    #sort the rows so that duplicate edges are adjacent to one another 
    edges_range_sorted = pd.DataFrame( edges_all_sorted  ).sort_values([0,1]).values
    edges_sorted = edges_range_sorted[:,0:2]

    #convert sorted indices to indices that correspond to face numbers
    #DEBUG commented out following line because it isnt' used:
    #sorted_indices = edges_range_sorted[:,2] % faces.shape[0]

    # the edges are reshuffled once by sorting them by row and then by extracting unique edges
    # we need to keep track of these transformations so that we can relate the shuffled edges to the 
    # faces they belong to.
    edges, edges_idx, counts = np.unique(edges_sorted , axis=0, return_index=True, return_counts=True)
    edges = edges.astype(np.uint32)

    assert np.sum(counts!=2) == 0,'Error: more than two faces per edge {}'.format( edges_sorted[edges_idx[counts!=2]])     
    #edge_range = np.arange(edges_all.shape[0]).astype(int) % faces.shape[0]
    return edges

def add_entry(d,i, lst):
    try :
        d[i] += lst
    except KeyError:
        d[i]=list(lst)
    return d[i]

def get_ngh_from_faces(faces):

    ngh={} 

    for count, (i,j,k) in enumerate(faces):
        ngh[i] = add_entry(ngh, int(i), [int(j),int(k)] )
        ngh[j] = add_entry(ngh, int(j), [int(i),int(k)] )
        ngh[k] = add_entry(ngh, int(k), [int(j),int(i)] )

    for key in range(len(ngh.keys())):
        ngh[int(key)] = list(np.unique(ngh[key]))

    return ngh

def upsample_over_faces(surf_fn, resolution, out_fn,  face_mask, coord_mask=None, profiles_vtr=None, slab_start=None, slab_end=None, new_points_only=False) :
    coords, faces, _ = load_mesh(surf_fn)

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
    
    # Choice 2 : if values are provided, interpolate these over the face, otherwise create 0-array
    if profiles_vtr != None :
        face_vertex_values = profiles_vtr[faces] 
        face_vertex_values = face_vertex_values[face_mask,:]
    else :
        face_vertex_values = np.zeros([face_coords.shape[0],3])

    ngh = get_ngh_from_faces(faces)

    points, values, new_points_gen = calculate_upsampled_points(faces, face_coords, face_vertex_values, resolution, new_points_only=new_points_only)

    np.savez(out_fn, points=points, values=values)
    print('\t\tSaved', out_fn)

    return points, values, new_points_gen

def find_neighbours(ngh, p, i, points, resolution, eps=0.001, target_nngh=6):
    counter = 1 
    #radius = resolution * counter + eps
    #idx0 = (points[:,0] <= p[0] + radius) & (points[:,0] >= p[0] - radius)
    #idx1 = (points[:,1] <= p[1] + radius) & (points[:,1] >= p[1] - radius)
    #idx2 = (points[:,2] <= p[2] + radius) & (points[:,2] >= p[2] - radius)

    #idx = idx0 & idx1 & idx2
    
    #if len(idx) > 6 :
    d=np.sqrt(np.sum(np.power(points - p,2),axis=1)) 
    d_idx = np.argsort(d).astype(int)[0:target_nngh]
    ngh = ngh[d_idx]

    return (i, ngh)

def create_point_blocks(points, resolution, scale=10):
    blocks = ((points - np.min(points, axis=0)) / (resolution*scale)).astype(np.uint16)
    n_blocks = np.max(blocks,axis=0).astype(int) +1
    return blocks, n_blocks



def find_neighbours_within_radius(ngh, nngh, points, blocks, n_blocks, resolution) :
    
    indices = np.arange(points.shape[0]).astype(int)
    n_total = np.product(n_blocks)
    counter=0

    for bx in range(n_blocks[0]):
        bx0=max(0,bx-1)
        bx1=min(n_blocks[0]-1,bx+1)
        
        bx_idx = bx==blocks[:,0]
        ngh_idx_0 = (blocks[:,0]>=bx0) & (blocks[:,0]<=bx1)
        

        for by in range(n_blocks[1]) :
            by0=max(0,by-1)
            by1=min(n_blocks[1]-1,by+1)

            by_idx = by==blocks[:,1]
            
            ngh_idx_1 = (blocks[:,1]>=by0) & (blocks[:,1]<=by1)

            for  bz in range(n_blocks[2]):
                if counter % 10 == 0 : print(np.round(100*counter/n_total,3), end='\r')

                bz0=max(0,bz-1)
                bz1=min(n_blocks[2]-1,bz+1)
                
                ngh_idx_2 = (blocks[:,2]>=bz0) & (blocks[:,2]<=bz1)

                ngh_idx = ngh_idx_0 * ngh_idx_1 * ngh_idx_2

                cur_idx = bx_idx & by_idx & (bz==blocks[:,2])
                
                cur_indices = indices[cur_idx ]
                cur_points = points[ cur_idx, : ]
                ngh_points = points[ ngh_idx, : ]
                block_ngh_list = Parallel(n_jobs=num_cores)(delayed(find_neighbours)(p, i,  ngh_points, indices[ngh_idx], resolution) for i, p in zip(cur_indices, cur_points) ) 
                 
                for key, item in block_ngh_list:
                    ngh[key]=item
                    nngh[key]=len(item)
                
                counter+=1

    return ngh, nngh

def link_points(points, ngh, resolution):
    print('\t\tLinking points to create a mesh.') 
    
    
    nngh=np.zeros(points.shape[0]).astype(np.uint8)

    block_ngh_list = Parallel(n_jobs=num_cores)(delayed(find_neighbours)(cur_ngh, points[i], i, points[cur_ngh], resolution) for i, cur_ngh in ngh.items() ) 
    
    for key, item in block_ngh_list:
        ngh[key]=list(np.unique(item))
    

    #print('Number of vertices with insufficient ngh', np.sum(nngh != 6))
    return ngh

def get_faces_from_neighbours(ngh):
    face_dict={}
    print('\tCreate Faces')
    for i in range(len(ngh.keys())):
        if i % 1000 : print(f'2. {100*i/ngh.shape[0]} %', end='\r')
        for ngh0 in ngh[i] :
            for ngh1 in ngh[ngh0] :
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

def mult_vector(v0,v1,x,y,p):
    mult = lambda a,b : np.multiply(np.repeat(a.reshape(a.shape[0],1),b.shape,axis=1), b).T
    w0=mult(v0,x)
    w1=mult(v1,y)
    # add the two vector components to create points within triangle
    p0 = p + w0 + w1 
    return p0

def interpolate_face(points, values, resolution, output=None, new_points_only=False):
    # calculate vector on triangle face
    v0, v1 = get_triangle_vectors(points)

    #calculate the magnitude of the vector and divide by the resolution to get number of 
    #points along edge
    calc_n = lambda v : np.ceil( np.sqrt(np.sum(np.power(v,2)))/resolution).astype(int)
    mag_0 = calc_n(v0)
    mag_1 = calc_n(v1)

    n0 = max(2,mag_0) #want at least the start and end points of the edge between two vertices
    n1 = max(2,mag_1)

    #calculate the spacing from 0 to 100% of the edge
    l0 = np.linspace(0,1,n0)
    l1 = np.linspace(0,1,n1)

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
    p0 = mult_vector(v0,v1,x,y,points[0,:])
    
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
        self.x = x
        self.y = y
    
    def generate_point(self, points) :
        cur_points = points[self.face]
        v0, v1 = get_triangle_vectors(cur_points)
        new_point = v0*self.x+ v1*self.y + points[0,:] 
        return new_point
        

def calculate_upsampled_points(faces,  face_coords, face_vertex_values, resolution, new_points_only=False):
    points=np.zeros([face_coords.shape[0]*5,3])
    values=np.zeros([face_coords.shape[0]*5])
    n_points=0
    new_points_gen = {}

    for f in range(face_coords.shape[0]):
        #if f % 1000 == 0 : print(f'\t\tUpsampling Faces: {100.*f/face_coords.shape[0]:.3}',end='\r')
        #check if it's worth upsampling the face
        coords_voxel_loc = np.unique(np.rint(face_coords[f]/resolution).astype(int), axis=0)
        
        #assert np.sum(face_vertex_values[f] == 0) == 0, f'Error: found 0 in vertex values for face {f}'

        if coords_voxel_loc.shape[0] > 1 :
            p0, v0, x, y = interpolate_face(face_coords[f], face_vertex_values[f], resolution, new_points_only=new_points_only)
        else : 
            p0 = face_coords[f]
            v0 = face_vertex_values[f]
        
        if n_points + p0.shape[0] >= points.shape[0]:
            points = np.concatenate([points,np.zeros([face_coords.shape[0],3])], axis=0)
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

def identify_target_edges_within_slab(edge_mask, section_numbers, ligand_vol_fn, coords, edges, resolution, ext='.nii.gz'):
    img = nb.load(ligand_vol_fn)
    ligand_vol = img.get_fdata()
    step = img.affine[1,1]
    start = img.affine[1,3]
    ydir = np.sign(step)
    resolution_step = resolution * ydir
        
    e0 = edges[:,0]
    e1 = edges[:,1]
    c0 = coords[e0,1]
    c1 = coords[e1,1]
   
    edge_range = np.arange(0, edges.shape[0]).astype(int)
    edge_range = edge_range[edge_mask == False]

    edge_y_coords = np.vstack([c0,c1]).T

    idx_1 = np.argsort(edge_y_coords,axis=1)
    edges = np.take_along_axis(edge_y_coords, idx_1, 1)
    sorted_edge_y_coords = np.take_along_axis(edge_y_coords, idx_1, 1)
    
    idx_0 = np.argsort(edges,axis=0)
    sorted_edge_y_coords = np.take_along_axis( sorted_edge_y_coords, idx_0, 0)
    edges = np.take_along_axis( edges, idx_0, 0)

    #ligand_y_profile = np.sum(ligand_vol,axis=(0,2))
    #section_numbers = np.where(ligand_y_profile > 0)[0]
    
    section_counter=0
    current_section_vox = section_numbers[section_counter]
    current_section_world = current_section_vox * step + start

    for i in edge_range :
        y0,y1 = sorted_edge_y_coords[i,:]
        e0,e1 = edges[i]

        crossing_edge = (y0 < current_section_world) & (y1 > current_section_world + resolution_step)

        start_in_edge = ((y0 >= current_section_world) & (y0 < current_section_world+resolution_step)) & (y1 > current_section_world + resolution_step)

        end_in_edge = (y0 < current_section_world) & ((y1>current_section_world) & (y1 <= current_section_world + resolution_step))
       
        if crossing_edge + start_in_edge + end_in_edge > 0 :
            edge_mask[i]=True

        if y0 > current_section_world :
            section_counter += 1
            if section_counter >= section_numbers.shape[0] :
                break
            current_section_vox = section_numbers[section_counter]
            current_section_world = current_section_vox * step + start
    
    return edge_mask 

def transform_surface_to_slabs( slab_dict, thickened_dict,  out_dir, surf_fn, ref_gii_fn=None, faces_fn=None, ext='.surf.gii'):
    surf_slab_space_dict={}

    for slab, curr_dict in slab_dict.items() :
        thickened_fn = thickened_dict[str(slab)]
        nl_3d_tfm_fn = slab_dict[str(slab)]['nl_3d_tfm_fn']
        surf_slab_space_fn = f'{out_dir}/slab-{slab}_{os.path.basename(surf_fn)}' 
        
        surf_slab_space_dict[slab] = {}
        surf_slab_space_dict[slab]['surf'] = surf_slab_space_fn
        surf_slab_space_dict[slab]['vol'] = thickened_fn
        print('\t-->',slab, surf_slab_space_fn)
        if not os.path.exists(surf_slab_space_fn) :
            apply_ants_transform_to_gii(surf_fn, [nl_3d_tfm_fn], surf_slab_space_fn, 0, faces_fn=faces_fn, ref_gii_fn=ref_gii_fn)
    return surf_slab_space_dict

def apply_ants_transform_to_gii( in_gii_fn, tfm_list, out_gii_fn, invert, ref_gii_fn=None, faces_fn=None):
    print("transforming", in_gii_fn)
    print("to", out_gii_fn)

    origin = [0,0,0]
    if type(ref_gii_fn) == type(None) :
        ref_gii_fn = in_gii_fn

    if os.path.splitext(ref_gii_fn)[1] in ['.pial', '.white'] : 
        _, _, volume_info = load_mesh(ref_gii_fn)
        origin = volume_info['cras'] 
    else : volume_info = ref_gii_fn

    ext = os.path.splitext(in_gii_fn)[1]
    if ext in ['.pial', '.white', '.gii'] : 
        coords, faces, _ = load_mesh(in_gii_fn)
    elif  ext == '.npz' :
        coords = np.load(in_gii_fn)['points']
        faces=None
    else :
        coords = h5.File(in_gii_fn)['data'][:]
        if os.path.splitext(faces_fn)[1] == '.h5' :
            faces_h5=h5.File(faces_fn,'r')
            faces = faces_h5['data'][:]
    
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
    #temp_out_fn=tempfile.NamedTemporaryFile().name+'.csv' #DEBUG
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
    #os.remove(temp_out_fn) DEBUG

    new_coords = df[['x','y','z']].values
    out_basename, out_ext = os.path.splitext(out_gii_fn)
    if out_ext == '.h5':
        f_h5 = h5.File(out_gii_fn, 'w')
        f_h5.create_dataset('data', data=new_coords) 
        f_h5.close()
        save_mesh(out_path+'.surf.gii', new_coords, faces, volume_info=volume_info)
    elif out_ext =='.npz' :
        np.savez(out_basename, points=coords)
    else :
        print('\tWriting Transformed Surface:',out_gii_fn, faces.shape )
        save_mesh(out_gii_fn, new_coords, faces, volume_info=volume_info)

    #obj_fn = out_path +  '.obj'
    #save_obj(obj_fn,coords, faces)

def get_section_intervals(vol):
    section_sums = np.sum(vol, axis=(0,2))
    valid_sections = section_sums > np.min(section_sums)
    labeled_sections, nlabels = label(valid_sections)
    if nlabels < 2:
        print('Error: there must be a gap between thickened sections. Use higher resolution volumes.')

    
    ##intervals = [ (np.where(labeled_sections==i)[0][0], np.where(labeled_sections==i)[0][-1]) for i in range(1, nlabels+1) ]
    
    intervals = [ (np.where(labeled_sections==i)[0][0], np.where(labeled_sections==i)[0][-1]+1) for i in range(1, nlabels+1) ]
    
    assert len(intervals) > 0 , 'Error: no valid intervals found for volume.'  
    return intervals
    
def resample_to_output(vol, aff, resolution_list, order=1, dtype=None):
    initial_vol_sum = np.sum(np.abs(vol))
    dim_range=range(len(vol.shape))
    calc_new_dim = lambda length, step, resolution : np.ceil( (length*abs(step))/resolution).astype(int)
    dim = [ calc_new_dim(vol.shape[i], aff[i,i], resolution_list[i]) for i in dim_range ]
    
    assert len(np.where(np.array(dim)<=0)[0]) == 0 , f'Error: dimensions <= 0 in {dim}'

    out_vol=np.zeros(dim,dtype=vol.dtype)
    

    vol_size_gb = out_vol.nbytes / 1000000000

    if vol_size_gb > 2 : 
        yscale = abs(aff[1,1])/resolution_list[1]
        n = np.ceil(vol_size_gb / 2).astype(int)
        step = np.rint(vol.shape[1]/n).astype(int)
        for y0 in range(0,vol.shape[1],step) :
            y1=min(vol.shape[1],y0+step)
            x0 = max(0, np.floor(y0*yscale).astype(int))
            x1 = min(out_vol.shape[1], np.ceil(y1*yscale).astype(int))
            temp_vol = resize(vol[:,y0:y1,:], [dim[0],x1-x0,dim[2]], order=order)
            
            out_vol[:,x0:x1,:] = temp_vol
            del temp_vol
    else :
        out_vol = resize(vol, dim, order=order )
    
    assert np.sum(np.abs(vol)) > 0  or initial_vol_sum == 0, 'Error: empty output volume after <resize> in <resample_to_output>'
    aff[dim_range,dim_range] = resolution_list
    return nib.Nifti1Image(out_vol, aff, dtype=dtype )


class AntsParams():
    def __init__(self, resolution_list, resolution, base_itr, max_resolution=None, start_resolution=None, verbose=False):

        if type(max_resolution) == type(None) :
            self.max_resolution=resolution
        else :
            self.max_resolution=max_resolution

        if start_resolution != None :
            resolution_list = [ r for r in resolution_list if float(r) < float(start_resolution) ]
        if len(resolution_list) == 0 : return None

        self.resolution_list=resolution_list
        self.max_n = len(resolution_list)
        print(resolution_list)
        self.cur_n = resolution_list.index(resolution)
        self.max_itr=int((self.max_n+1)*base_itr)
        
        self.f_list = self.gen_downsample_factor_list(resolution_list)
        self.max_downsample_factor = int(self.f_list[0] )

        self.f_str = self.get_downsample_params(resolution_list)
        self.s_list = self.gen_smoothing_factor_list(resolution_list)
        self.s_str = self.get_smoothing_params(resolution_list)
        self.itr_str =  self.gen_itr_str(self.max_itr, base_itr)

        if verbose: self._print()

        assert len(self.f_list) == len(self.s_list) == len(self.itr_str.split('x')), f'Error: incorrect number of elements in: \n{self.f_list}\n{self.s_list}\n{self.itr_str}\n{resolution_list}'
    def _print(self):
        print('Iterations:\t',self.itr_str)
        print('Factors:\t', self.f_str)
        print('Smoothing:\t',self.s_str)

    def gen_itr_str(self,max_itr, step):
        itr_str = 'x'.join([str(int(max_itr - i*step)) for i in range(self.cur_n+1) if self.resolution_list[i] >= self.max_resolution ])
        itr_str ='['+ itr_str +',1e-7,20 ]' 
        return itr_str

    def gen_smoothing_factor_string(self, lst):
        return 'x'.join( [str(i) for i in lst] ) + 'vox'

    def gen_smoothing_factor_list(self, resolution_list):
        #smooth_f = lambda x, y:  np.round(float(float(x)/float(y))/np.pi,2)
        smooth_f = lambda x, y:  np.round(float(float(x)/float(y)-1)*0.2,3)
        #smooth_f = lambda x, y:  (float(x)/float(y))/2
        return [ smooth_f(resolution_list[i],resolution_list[self.cur_n]) if i != self.cur_n else 0 for i in range(self.cur_n+1) if self.resolution_list[i] >= self.max_resolution ] 

    def get_smoothing_params(self, resolution_list):
        s_list = self.gen_smoothing_factor_list(resolution_list)
        return self.gen_smoothing_factor_string(s_list)

    def calc_downsample_factor(self, cur_res, image_res):
        return np.rint(float(cur_res)/float(image_res)).astype(int).astype(str) #DEBUG
        #return np.floor(1+np.log2(float(cur_res)/float(image_res))).astype(int).astype(str)

    def gen_downsample_factor_list(self, resolution_list) :
        factors = [ self.calc_downsample_factor(resolution_list[i], resolution_list[self.cur_n]) for i in range(self.cur_n+1) if self.resolution_list[i] >= self.max_resolution  ]
        return factors

    def get_downsample_params(self, resolution_list) :
        return 'x'.join( self.gen_downsample_factor_list(resolution_list) )


def check_transformation_not_empty(in_fn, ref_fn, tfm_fn, out_fn, empty_ok=False):
    assert os.path.exists(out_fn), f'Error: transformed file does not exist {out_fn}'
    assert np.sum(np.abs(nib.load(out_fn).dataobj)) > 0 or empty_ok, f'Error in applying transformation: \n\t-i {in_fn}\n\t-r {ref_fn}\n\t-t {tfm_fn}\n\t-o {out_fn}\n'

def simple_ants_apply_tfm(in_fn, ref_fn, tfm_fn, out_fn,ndim=3,n='Linear',empty_ok=False):
    if not os.path.exists(out_fn):
        str0 = f'antsApplyTransforms -v 0 -d {ndim} -i {in_fn} -r {ref_fn} -t {tfm_fn}  -o {out_fn}'
        shell(str0, verbose=True)
        check_transformation_not_empty(in_fn, ref_fn, tfm_fn, out_fn,empty_ok=empty_ok)


def resample_to_autoradiograph_sections(brain, hemi, slab, resolution,input_fn, ref_fn, tfm_inv_fn, iso_output_fn, output_fn):
    '''
    About:
        Apply 3d transformation and resample volume into the same coordinate space as 3d receptor volume.           This produces a volume with 0.02mm dimension size along the y axis.

    Inputs:
        brain:      current subject brain id
        hemi:       current hemisphere (R or L)
        slab:       current slab number
        resolution:     current resolution level
        input_fn:     gm super-resolution volume (srv) extracted from donor brain
        ref_fn:     brain mask of segmented autoradiographs
        tfm_inv_fn:   3d transformation from mni to receptor coordinate space
        srv_space_rec_fn:      
        
    Outpus:
        None
    '''
    simple_ants_apply_tfm(input_fn, ref_fn, tfm_inv_fn, '/tmp/tmp.nii.gz',ndim=3)
    
    img = nib.load('/tmp/tmp.nii.gz')
    vol = img.get_fdata()
    assert np.sum(vol) > 0, f'Error: empty volume {iso_output_fn}'
    
    aff = img.affine.copy()
    
    vol = ( 255*(vol-vol.min())/(vol.max()-vol.min()) ).astype(np.uint8)

    img_iso = resample_to_output(vol, aff, [float(resolution)]*3, order=0, dtype=np.uint8)
    img_iso.to_filename(iso_output_fn)
    
    aff = img.affine.copy()

    img3 = resample_to_output(vol, aff, [float(resolution),0.02, float(resolution)], order=1, dtype=np.uint8)
    
    img3.to_filename(output_fn)

    os.remove('/tmp/tmp.nii.gz')


def fix_affine(fn):
    try :
        ants.image_read(fn)
    except RuntimeError :
        img = nib.load(fn)
        dim = vol.shape
        aff = img.affine

        origin = list(affine[ [0,1],[3,3] ])
        spacing = list( affine[ [0,1],[0,1] ])
        ants_image = ants.from_numpy(img.get_fdata(), origin=origin, spacing=spacing)
        ants_image.to_filename(fn)
        try :
            ants.image_read(fn)
        except RuntimeError :
            print("Error could not fix direction cosines for"+fn)
            exit(1)

def read_points(fn, ndim = 3):
    start_read=False
    points0 = []
    points1 = []
    fn_list = []
    with open(fn,'r') as F:
        for line in F.readlines() :
            if start_read : 
                string_split = line.split(' ')
                points0 += [[ float(i) for i in string_split[1:(1+ndim)] ]]
                points1 += [[ float(i) for i in string_split[4:(4+ndim)] ]]

            if 'Points' in line : start_read=True
            
            if '%Volume:' in line : 
                fn = line.split(' ')[1]
                fn_list.append( os.path.basename(fn.rstrip()))
   
    return np.array(points0), np.array(points1), fn_list[0], fn_list[1]

def safe_ants_image_read(fn, tol=0.001, clobber=False):
    img = nib.load(fn)
    affine = img.affine
    origin = list(affine[ [0,1,2],[3,3,3] ])
    spacing = list( affine[ [0,1,2],[0,1,2] ])
    ants_image = ants.from_numpy(img.get_fdata(), origin=origin, spacing=spacing)
    return ants_imag


def points2tfm(points_fn, affine_fn, fixed_fn, moving_fn, ndim=3, transform_type="Affine", invert=False, clobber=False):

    #comFixed=[] 
    #comMoving=[] 
    #comMoving[-1] *= 1 
    if not os.path.exists(affine_fn) or clobber :

        fixed_img = ants.image_read(fixed_fn)
        moving_img = ants.image_read(moving_fn)


        comFixed = list( get_center_of_mass(fixed_img) ) 
        comMoving = list( get_center_of_mass(moving_img) )

        fixed_dirs = fixed_img.direction[[0,1,2],[0,1,2]]
        moving_dirs = moving_img.direction[[0,1,2],[0,1,2]]
        # f=rec_points / m=mni_points
        fixed_points, moving_points, fixed_fn, moving_fn = read_points(points_fn, ndim=ndim)

        print('\t: Calculate affine matrix from points')
        if invert:
            # f=mni_points / m=rec_points
            temp_points = np.copy(moving_points)
            moving_points = np.copy(fixed_points)
            fixed_points = temp_points

        fixed_points = fixed_dirs * fixed_points
        moving_points = moving_dirs * moving_points

        fixed_points = fixed_points - np.mean(fixed_points,axis=0) + comFixed
        moving_points = moving_points - np.mean(moving_points,axis=0) + comMoving

        landmark_tfm = fit_transform_to_paired_points(moving_points, fixed_points, transform_type=transform_type , centerX=comFixed, centerY=comMoving)

        ants.write_transform(landmark_tfm, affine_fn)
        
        df=pd.DataFrame( {'x':moving_points[:,0],'y':moving_points[:,1],'z':moving_points[:,2]} )
        
        rsl_points = ants.apply_transforms_to_points(3, df, affine_fn, whichtoinvert=[True] )
        
        error = np.sum(np.sqrt(np.sum(np.power((rsl_points - fixed_points), 2), axis=1)))

    return affine_fn

def newer_than(input_list, output_list) :
    '''
    check if the files in input list are newer than target_filename
    '''
    for output_filename in output_list :
        t0 = os.path.getctime(output_filename)

        for input_filename in input_list :
            t1 = os.path.getctime(input_filename)

            if t1 < t0 : 
                print('Input file is newer than output file.')
                print('\tInput file:')
                print('\t',t1, time.ctime(os.path.getctime(input_filename)))
                print('\tOutput file:')
                print('\t',t0, time.ctime(os.path.getctime(output_filename)))
                exit(1)
                return True
    
    return False

def run_stage(input_list, output_list):
    '''
    check if stage needs to be run
    '''
    # check if inputs exist
    for input_filename in input_list :
        assert os.path.exists(input_filename), f'Error: input to stage does not exist, {input_filename}'
    
    # check if outputs exist
    for output_filename in output_list :
        if not os.path.exists(output_filename) :
            return True

    # check if inputs are newer than existing outputs
    #return newer_than(input_list, output_list)
    return False


def get_seg_fn(dirname, y, resolution, filename, suffix=''):
    filename = re.sub('.nii.gz',f'_y-{int(y)}_{resolution}mm{suffix}.nii.gz', os.path.basename(filename))
    return '{}/{}'.format(dirname, filename )

def gen_2d_fn(prefix,suffix,ext='.nii.gz'):
    return f'{prefix}{suffix}{ext}'

def save_sections(file_list, vol, aff, dtype=None) :
    xstep = aff[0,0]
    ystep = aff[1,1]
    zstep = aff[2,2]

    xstart = aff[0,3]
    zstart = aff[2,3]

    for fn, y in file_list:
        affine = np.array([  [xstep,  0, 0, xstart ],
                                [0, zstep, 0, zstart ],
                                [0, 0,  0.02, 0 ],
                                [0, 0,  0, 1]])
        i=0
        if np.sum(vol[:,int(y),:]) == 0 :
            # Create 2D srv section
            # this little while loop thing is so that if we go beyond  brain tissue in vol,
            # we find the closest y segement in vol with brain tissue
            while np.sum(vol[:,int(y-i),:]) == 0 :
                i += (ystep/np.abs(ystep)) * 1
        sec = vol[ :, int(y-i), : ]
        nib.Nifti1Image(sec , affine, dtype=dtype).to_filename(fn)

def get_to_do_list(df,out_dir,str_var,ext='.nii.gz'):
    to_do_list=[]
    for idx, (i, row) in enumerate(df.iterrows()):
        y = int(row['slab_order'])
        assert int(y) >= 0, f'Error: negative y value found {y}'
        prefix = f'{out_dir}/y-{y}' 
        fn = gen_2d_fn(prefix,str_var,ext=ext)
        if not os.path.exists(fn) : to_do_list.append( [fn, y])
    return to_do_list

def create_2d_sections( df,  srv_fn, resolution, output_dir, dtype=None, clobber=False) :
    fx_to_do=[]
    
    tfm_dir = output_dir + os.sep + 'tfm'
    os.makedirs(tfm_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    fx_to_do = get_to_do_list(df, tfm_dir, '_fx') 

    if len( fx_to_do ) > 0 :
        print('srv_fn: ', srv_fn)
        srv_img = nib.load(srv_fn)
        affine = srv_img.affine
        srv = srv_img.get_fdata()
        save_sections(fx_to_do, srv, affine, dtype=dtype)
        

def w2v(i, step, start):
    return np.round( (i-start)/step ).astype(int)

def v2w(i, step, start) :
    return start + i * step   
def splitext(s):
    try :
        ssplit = os.path.basename(s).split('.')
        ext='.'+'.'.join(ssplit[1:])
        basepath= re.sub(ext,'',s)
        return [basepath, ext]
    except TypeError :  
        return s
def setup_tiers(df, tiers_str):
    tiers_list = [ j.split(",") for j in tiers_str.split(";")]
    i=1
    df["tier"]=[0] * df.shape[0]
    if tiers_str != '' :
        for tier in tiers_list :
            for ligand in tier :
                df["tier"].loc[ df.ligand == ligand ] = i
            
            i += 1
        df=df.loc[df["tier"] != 0 ]
    return(df)

def pad_size(d):
    if d % 2 == 0 :
        pad = (int(d/2), int(d/2))
    else :
        pad = ( int((d-1)/2), int((d+1)/2))
    return(pad)
    

def add_padding(img, zmax, xmax):
    z=img.shape[0]
    x=img.shape[1]
    
    dz = zmax - z
    dx = xmax - x
    
    z_pad = pad_size(dz)
    x_pad = pad_size(dx)
    
    img_pad = np.pad(img, (z_pad,x_pad), 'constant', constant_values=np.max(img))
    #img_pad = np.zeros([zmax,xmax])
    #img_pad[z_pad[0]:, x_pad[0]: ] = img
    return img_pad


def gm_segment(img):
    mid = np.mean(img[img>0])
    upper=np.max(img)
    if upper > 0 :
        init=np.array([0,mid,upper]).reshape(-1,1)
        cls = KMeans(3, init=init).fit_predict(img.reshape(-1,1)).reshape(img.shape)
        cls[ cls != 2 ] = 0
        return cls.astype(float)
    return img

newlines = ['\n', '\r\n', '\r']
def unbuffered(proc, stream='stdout'):
    stream = getattr(proc, stream)
    with contextlib.closing(stream):
        while True:
            out = []
            last = stream.read(1)
            # Don't loop forever
            if last == '' and proc.poll() is not None:
                break
            while last not in newlines:
                # Don't loop forever
                if last == '' and proc.poll() is not None:
                    break
                out.append(last)
                last = stream.read(1)
            out = ''.join(out)
            print(out)
            yield out

def shell(cmd, verbose=False,exit_on_failure=True):
    '''Run command in shell and read STDOUT, STDERR and the error code'''
    stdout=""
    if verbose :
        print(cmd)


    process=Popen( cmd, shell=True, stdout=PIPE, stderr=STDOUT, universal_newlines=True, )

    for line in unbuffered(process):
        stdout = stdout + line + "\n"
        if verbose :
            print(line)

    errorcode = process.returncode
    stderr=stdout
    if errorcode != 0:
        print ("Error:")
        print ("Command:", cmd)
        print ("Stderr:", stdout)
        if exit_on_failure : exit(errorcode)
    return stdout, stderr, errorcode

def get_z_x_max(source_files):
    zmax=xmax=0
    for i in range(len(source_files)) : 
        f=source_files[i]
        fn = output_dir + os.sep + os.path.basename(f)
        if not os.path.exists(fn) :
            img = imageio.imread(f)
            img = downsample(img, fn,  step=0.2)
        else : 
            img = imageio.imread(fn)
        z=img.shape[1]
        x=img.shape[0]
        if x > xmax : xmax=x 
        if z > zmax : zmax=z 
        source_files[i] = fn

from scipy.ndimage import zoom


def safe_imread(fn) :
    img = imageio.imread(fn)
    if len(img.shape) > 2 :
        img = np.mean(img,axis=2)
    return img

    
def downsample(img, subject_fn="", step=0.2, raw_step=0.02, interp=3):
    #Calculate length of image based on assumption that pixels are 0.02 x 0.02 microns
    l0 = img.shape[0] * raw_step 
    l1 = img.shape[1] * raw_step

    #Calculate the length for the downsampled image
    dim0=int(np.ceil(l0 / step))
    dim1=int(np.ceil(l1 / step))

    #Calculate the standard deviation based on a FWHM (pixel step size) of downsampled image
    
    #Downsample
    img_dwn = resize(img.astype(float), (dim0, dim1), order=int(interp) ) 
     
    if subject_fn != "" : 
        imageio.imsave(subject_fn, img_dwn.astype(np.uint16))

    return(img_dwn)


def downsample_and_crop(source_lin_dir, lin_dwn_dir,crop_dir, affine, step=0.2, clobber=False):

    for f in glob(source_lin_dir+"/*.TIF") :
        dwn_fn = lin_dwn_dir + splitext(basename(f))[0] + '.nii.gz'
        if not os.path.exists(dwn_fn) or clobber :
            try :
                base=sub('#L','',basename(splitext(f)[0]))
                path_string=crop_dir+"/detailed/**/"+base+"_bounding_box.png"
                print("path:", path_string)

                crop_fn = glob(path_string)[0]
            except IndexError :
                print("\t\tDownsample & Crop : Skipping ", f)
                continue
            img = imageio.imread(f)
            if len(img.shape) == 3 : img = np.mean(img,axis=2)

            bounding_box = imageio.imread(crop_fn) 
            if np.max(bounding_box) == 0 : 
                bounding_box = np.ones(img.shape)
            else : 
                bounding_box = bounding_box / np.max(bounding_box)
            img = img * bounding_box 
            resample_to_output(img, affine, [step]*len(img.shape), order=5).to_filename(dwn_fn)
            print("downsampled filename", dwn_fn)

def world_center_of_mass(vol, affine):
    affine = np.array(affine)
    ndim = len(vol.shape)
    r = np.arange(ndim).astype(int)
    com = center_of_mass(vol)
    steps = affine[r,r]
    starts = affine[r,3]
    wcom = com * steps + starts
    return wcom

#def reshape_to_min_dim(vol):
from scipy.ndimage import shift

def recenter(vol, affine, direction=np.array([1,1,-1])):
    affine = np.array(affine)
    
    vol_sum_1=np.sum(np.abs(vol))
    assert vol_sum_1 > 0, 'Error: input volume sum is 0 in recenter'
    #wcom1 = world_center_of_mass(vol,affine) 

    ndim = len(vol.shape)
    vol[pd.isnull(vol)] = 0
    coi = np.array(vol.shape) / 2
    com = center_of_mass(vol)
    d_vox = np.rint(coi - com)
    d_vox[1]=0
    d_world = d_vox * affine[range(ndim),range(ndim)]
    d_world *= direction 
    affine[range(ndim),3] -= d_world
    
    print('\tShift in Segmented Volume by:', d_vox)
    vol = shift(vol,d_vox, order=0)
    #nib.Nifti1Image(vol, affine).to_filename('test_shifted.nii.gz')
    
    #wcom2 = world_center_of_mass(vol,affine) 
    steps = affine[range(ndim),range(ndim)]
    #assert np.sum(np.abs(wcom1- wcom2)) < np.sum(np.abs(steps))*3, f'Error: center of mass does not match {wcom1}, {wcom2}'
    #assert np.abs(vol_sum_1 - np.sum(vol))/vol_sum_1 < 0.02, f'Error: total intensities does not match after recentering'
    return vol, affine


def prefilter_and_downsample(input_filename, new_resolution, output_filename, 
                            new_starts=[None, None, None], recenter_image=False, dtype=None ):

    #img = nib.load(input_filename)
    img = ants.image_read(input_filename)

    
    if type(dtype) == type(None) : dtype=img.dtype
    
    direction = img.direction

    vol = img.numpy()
    assert np.sum(np.abs(vol)) > 0 , 'Error: empty input file for prefilter_and_downsample\n'+input_filename
    ndim = len(vol.shape)

    affine = np.eye(4,4) #img.affine
    dim_range=range(ndim)
    affine[dim_range,dim_range]=img.spacing
    affine[dim_range,3]=img.origin

    if recenter_image : 
        vol, affine = recenter(vol,affine)

    new_affine = np.copy( affine )
    for i, new_start in enumerate(new_starts) :
        if new_start != None : 
            new_affine[3,i] = float( new_start )
    
    new_resolution = np.array(new_resolution).astype(float)

    if ndim ==3 : 
        if vol.shape[2] == 1 : vol = vol.reshape([vol.shape[0],vol.shape[1]])
    
    if ndim == 2 :
        new_affine[0,0] = new_resolution[0]
        new_affine[1,1] = new_resolution[1]
        
        steps = np.array( [ affine[0,0], affine[1,1] ] )

    elif ndim == 3 :
        steps = np.array( [ affine[0,0], affine[1,1], affine[2,2] ] )
        new_affine[0,0] = new_resolution[0]
        new_affine[1,1] = new_resolution[1]
        new_affine[2,2] = new_resolution[2]
    else :
        print(f'Error: number of dimensions ({ndim}) does not equal 2 or 3 for {input_filename}')
        exit(1)


    vol = resample_to_output( vol, affine, new_resolution, order=5).get_fdata()
    new_dims = [ vol.shape[0], vol.shape[1] ]

    if len(vol.shape) == 3 :
        if vol.shape[2] != 1 :
            new_dims.append(vol.shape[2])

    # Warning: This reshape step is absolutely necessary to correctly apply ants transforms.
    #           nibabel's resampling function will change the dimensions from, say, (x,y) to (x,y,1)
    #           This throws things off for ants so the volume has to be reshaped back to original dimensions.
    vol = vol.reshape(*new_dims) 
    assert np.sum(np.abs(vol)) > 0, 'Error: empty output array for prefilter_and_downsample\n'+output_filename
    nib.Nifti1Image(vol, new_affine,  direction=direction, dtype=dtype).to_filename(output_filename)

    return vol

    return vol


def rgb2gray(rgb): return np.mean(rgb, axis=2)

def find_min_max(seg):
    m = np.max(seg)
    fmin = lambda a : min(np.array(range(len(a)))[a == m])
    fmax = lambda a : max(np.array(range(len(a)))[a == m])
    
    xmax = [ fmax(seg[i,:]) for i in range(seg.shape[0]) if np.sum(seg[i,:]==m) > 0 ]
    xmin = [ fmin(seg[i,:]) for i in range(seg.shape[0]) if np.sum(seg[i,:]==m) > 0 ]
    
    y_i = [ i for i in range(seg.shape[0]) if np.sum(seg[i,:]) != 0  ]

    ymax = [ fmax(seg[:,i]) for i in range(seg.shape[1]) if np.sum(seg[:,i]==m) > 0 ]
    ymin = [ fmin(seg[:,i]) for i in range(seg.shape[1]) if np.sum(seg[:,i]==m) > 0 ]
    x_i =  [ i for i in range(seg.shape[1]) if np.sum(seg[:,i]) != 0 ]

    if xmin == [] : xmin = [ 0 ]
    if xmax == [] : xmax = [ seg.shape[1] ]
    if  y_i == [] : y_i = [0]

    if ymin == [] : ymin = [ 0 ]
    if ymax == [] : ymax = [ seg.shape[0] ]
    if  x_i == [] : x_i = [0]
    
    return ymin, ymax, xmin, xmax, y_i, x_i

def extrema(labels) :
    xx, yy = np.meshgrid(range(labels.shape[1]), range(labels.shape[0]))
    xxVals = (xx[labels > 0])
    yyVals = (yy[labels > 0])
    y0 = min(yyVals)
    y1 = max(yyVals) 
    x0 = min(xxVals)
    x1 = max(xxVals)
    return([x0,y0,x1,y1])



def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2 ,'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        #hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]
        #hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[ np.round(src[r,c]).astype(int)] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, 256): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst


def split_filename(fn):
    dfout.index = dfout.index.map(lambda x : re.sub(r"([0-9])s([0-9])", r"\1#\2", x, flags=re.IGNORECASE))
    ar=list(map(lambda x: re.split('#|\.|\/',basename(x)), dfout.index.values))
    df0=pd.DataFrame(ar, columns=["mri","slab","hemisphere","ligand","sheat","repeat"])
    df0.index=dfout.index
    dfout=pd.concat([df0, dfout], axis=1)
    return dfout

def sort_slices(df, slice_order) :
    df["order"]=[-1]*df.shape[0]
    for fn2, df2 in slice_order.groupby("name"):
        fn2b = re.sub(r"([0-9])s([0-9])", r"\1#\2", fn2, flags=re.IGNORECASE)
        fn2c = re.split('#|\.|\/|\_',fn2b)
        if len(fn2c) < 2 : continue
        mri=fn2c[2] 
        slab=fn2c[3]
        hemisphere=fn2c[4]
        ligand=fn2c[5]
        sheet=fn2c[6]
        repeat=fn2c[7]
        df["order"].loc[ (mri == df.mri) & (slab == df.slab) & (hemisphere == df.hemisphere) & (ligand == df.ligand) & (sheet == df.sheet) & (repeat == df.repeat ) ] = df2["global_order"].values[0]
    df.sort_values(by=["order"], inplace=True)
    return df



def set_slice_order(slice_order_fn="") :
    slice_order_list=[]
    for i in range(1,7) : #FIXME : Should not be hardcoded
        slice_order_fn = "MR1_R_slab_"+str(i)+"_section_numbers.csv"
        if os.path.exists(slice_order_fn) : 
            df0=pd.read_csv(slice_order_fn)
            df0["slab"]= [i] * df0.shape[0]
            slice_order_list.append(df0)
    slice_order = pd.concat(slice_order_list)
    slice_order["global_order"] =slice_order["number"].astype(int)

    slice_order_unique = np.sort(slice_order["slab"].unique())
    for i in slice_order_unique[1:] :
        prev_slab = i - 1
        prev_slab_max = slice_order["global_order"].loc[ slice_order["slab"] == prev_slab ].max() 
        slice_order["global_order"].loc[ slice_order["slab"] == i ] += prev_slab_max

    return slice_order

def set_slice_name(source_files, cols, include_str, exclude_str):
    include_list = include_str.split(',')
    exclude_list = exclude_str.split(',')
    df=pd.DataFrame(columns=cols)
    df_list=[] 
    for f0 in source_files:
        f=re.sub(r"([0-9])s([0-9])", r"\1#\2", f0, flags=re.IGNORECASE)
        ar=re.split('#|\.|\/|\_',basename(f))
        
        ar=[f0]+ar
        processing_str=""
        ar_enum = enumerate(ar)
        n_cols = len(cols)
        n_ar = len(ar)
        if n_ar  > n_cols :
            k = n_ar - n_cols + 1
            ar_short = ar[(n_cols-k):(n_ar-1)]
            
            processing_string = '-'.join(ar_short)

            ar = ar[0:(len(cols)-k)] + [processing_string] + [ar[-1]]
        
        if len(ar) != len(cols) : 
            print("Warning! Skipping :")
            print(ar)
            continue    
        df0=pd.DataFrame([ar], columns=cols)
        ligand = df0.ligand[0]
        if not include_str == '': 
            if not ligand in include_list : continue
        if not exclude_str == '': 
            if ligand in exclude_list : 
                continue
        df_list.append(df0)
        #n = f_split[6].split("_")[0]
        #df = df.append( pd.DataFrame([[f0, mr, slab, hemi, tracer,n]], columns=cols))
    df=pd.concat(df_list)
    return df

def set_slab_border(df) :
    df["border"] = [0] * df.shape[0]
    for slab, df0 in df.groupby("slab") :
        min_order = df0["order"].min() + df0["order"].max() * 0.02
        max_order = df0["order"].max() * 0.98
        df0["border"].loc[ (df0["order"] < min_order ) | ( df0["order"] > max_order)  ] = 1
        df.loc[ df["slab"] == slab ] = df0

    return df
    
def set_csv(source_files, output_dir, include_str="", exclude_str="", slice_order_fn="", out_fn="receptor_slices.csv", clobber=False, df_with_order=False):
    
    if not os.path.exists(output_dir) : os.makedirs(output_dir)
    if not os.path.exists(output_dir+os.sep+out_fn ) or clobber:
        #Load csv files with order numbers
        cols=["filename", "a","b","mri","slab","hemisphere","ligand","sheet","repeat", "processing","ext"] 
        if df_with_order :
            cols=["filename","order", "a","b","mri","slab","hemisphere","ligand","sheet","repeat", "processing","ext"] 
        slice_order = set_slice_order()
        df = set_slice_name(source_files, cols, include_str, exclude_str)
        df = sort_slices(df, slice_order )
        ###df = set_slab_border(df)
        ###if os.path.exists(slice_order_fn)  :
        ###else : df.sort_values(by=["mri","slab","ligand","sheet","repeat"], inplace=True)
        if output_dir != "" : 
            df.to_csv(output_dir+os.sep+out_fn)
    else :
        df = pd.read_csv(output_dir+os.sep+out_fn)

    return(df)


