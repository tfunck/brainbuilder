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

os_info = os.uname()
global num_cores
if os_info[1] == 'imenb079':
    num_cores = 1 
else :
    num_cores = min(14, multiprocessing.cpu_count() )

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


def concatenate_points(to_do_list, all_points_npy):

    all_points = np.array([])
    all_values = np.array([])
    for _, out_fn in to_do_list:
        np_file = np.load(out_fn+'.npz')
        points = np_file['points']
        values = np_file['values']
        assert points.shape[0] == values.shape[0], f'Error: points ({points.shape}) and values ({values.shape}) dont match'
        if all_points.shape == (0,) :
            all_points=points
            all_values=values
        else :
            all_points = np.concatenate([all_points,points],axis=0)
            all_values = np.concatenate([all_values,values],axis=0)
        
        print('shape', points.shape, values.shape)

    np.savez(all_points_npy, points=all_points, values=all_values)
    return all_points, all_values


def multi_mesh_to_volume(profiles, surf_depth_slab_dict, depth_list, dimensions, starts, steps, resolution, y0, y1, origin=[0,0,0], ref_fn=None):
    all_points=[]
    all_values=[]
    interp_vol = np.zeros(dimensions)
    n_vol = np.zeros_like(interp_vol)

    slab_start = min(y0,y1)
    slab_end = max(y0,y1)

    #if type(ref_fn) != None :
    #    print('Reference faces:', ref_fn)
    #    _, ref_faces = load_mesh_ext(ref_fn)
    #else : 
    #    ref_faces = None

    #to_do_list=[]
    for ii in range(profiles.shape[1]) :
        surf_fn = get_surf_from_dict(surf_depth_slab_dict[depth_list[ii]]) 
        print('\tSURF', surf_fn)

        if 'npz' in os.path.splitext(surf_fn)[-1] : ext = '.npz'
        else : ext='.surf.gii'
        #out_fn = re.sub(ext, f'_upsampled-face-{ii}', surf_fn)
        #to_do_list.append((surf_fn,out_fn))
        #to_do_ist.append((surf_fn,surf_fn))
        points = np.load(surf_fn)['points']
        assert points.shape[0] == profiles.shape[0], 'Error mismatch in number of points between {surf_fn} and vertex values file'
        interp_vol, n_vol = mesh_to_volume(points, profiles[:,ii], dimensions, starts, steps, interp_vol=interp_vol, n_vol=n_vol)

    #Parallel(n_jobs=num_cores)(delayed(upsample_over_faces)(surf_fn, resolution, out_fn, profiles_vtr=profiles[:,ii], slab_start=slab_start, slab_end=slab_end, ref_faces=ref_faces) for ii, (surf_fn, out_fn) in enumerate(to_do_list) ) 

    #all_points_npy = re.sub('.surf.gii', f'upsampled-face-all', surf_fn)
    #all_points, all_values = concatenate_points(to_do_list, all_points_npy)

    #print(f'\tDepths completed: {100*ii/profiles.shape[1]:.3}',end='\r')
        
    #np_file = np.load(out_fn+'.npz')
    #points = np_file['points']
    #values = np_file['values']
    interp_vol[ n_vol>0 ] = interp_vol[n_vol>0] / n_vol[n_vol>0]
    
    assert np.sum(interp_vol) != 0 , 'Error: interpolated volume is empty'
    return interp_vol



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

    #print('2.', np.sum( np.sum(edges_all==(26, 22251),axis=1)==2 ) ) 
    
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


def unique_points(points, scale=1000000000):
    #rpoints = np.rint(points * scale).astype(np.int64)
    upoints, unique_index, unique_inverse = np.unique(points.astype(np.float128).round(decimals=3),axis=0,return_index=True, return_inverse=True)

    return points[unique_index,:], unique_index, unique_inverse

def upsample_over_faces(surf_fn, resolution, out_fn,  face_mask=None, coord_mask=None, profiles_vtr=None, slab_start=None, slab_end=None, ref_faces=None) :
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

    #ngh = get_ngh_from_faces(faces)
    points, values, new_points_gen = calculate_upsampled_points(faces, face_coords, face_vertex_values, resolution)

    assert points.shape[1]==3, 'Error: shape of points is incorrect ' + points.shape 
    points, unique_index, unique_reverse = unique_points(points)
    #a=3114460
    #b=3114447
    #print(points[a], points[b])
    #print(np.linalg.norm(points[a]- points[b]) )

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
    print(d_idx)
    print(ngh)
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
        #check if it's worth upsampling the face
        #coords_voxel_loc = np.unique(np.rint(face_coords[f]/resolution).astype(int), axis=0)
        
        #assert np.sum(face_vertex_values[f] == 0) == 0, f'Error: found 0 in vertex values for face {f}'
        
        #if coords_voxel_loc.shape[0] > 1 :
        p0, v0, x, y = interpolate_face(face_coords[f], face_vertex_values[f], resolution*0.9, new_points_only=new_points_only)
        #else : 
        #   p0 = face_coords[f]
        #    v0 = face_vertex_values[f]
        
        if n_points + p0.shape[0] >= points.shape[0]:
            points = np.concatenate([points,np.zeros([face_coords.shape[0],3]).astype(np.float128)], axis=0)
            values = np.concatenate([values,np.zeros(face_coords.shape[0])], axis=0)
       
        new_indices = n_points + np.arange(p0.shape[0]).astype(int)
        cur_faces = faces[f]

        #Check if x and y pairs are unique
        #xy = np.column_stack([x,y])
        #if xy.shape[0] != np.unique(xy,axis=0).shape[0]:
        #    print(xy)
        #    exit(0)

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
        print('\tFROM:', surf_fn)
        print('\tTO:', surf_slab_space_fn)
        print('\tWITH:', nl_3d_tfm_fn)
        if not os.path.exists(surf_slab_space_fn) :
            apply_ants_transform_to_gii(surf_fn, [nl_3d_tfm_fn], surf_slab_space_fn, 0, faces_fn=faces_fn, ref_gii_fn=ref_gii_fn, ref_vol_fn=thickened_fn)

    return surf_slab_space_dict


def load_mesh_ext(in_fn, faces_fn=''):
    ext = os.path.splitext(in_fn)[1]
    faces=None
    volume_info = None

    if ext in ['.pial', '.white', '.gii', '.sphere', '.inflated'] : 
        coords, faces, volume_info = load_mesh(in_fn,correct_offset=True)
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

