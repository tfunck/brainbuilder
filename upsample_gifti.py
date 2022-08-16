import numpy as np
import os
import time
import sys
import nibabel as nb
#import utils.ants_nibabel as nib
import matplotlib.pyplot as plt
import h5py as h5
import tempfile
import argparse
import h5py
import tracemalloc
import multiprocessing
import linecache
import psutil 
import io
import utils.ants_nibabel as nib
from joblib import Parallel, delayed
from nibabel import freesurfer
from guppy import hpy
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from utils.mesh_io import save_mesh, load_mesh, load_mesh_geometry, save_obj, read_obj
from re import sub
from utils.utils import shell,splitext, get_edges_from_faces
from glob import glob
from sys import getsizeof
from time import time

global ram_counter
global ram_fn
ram_counter=0
ram_fn='/tmp/tmp.csv'

if os.path.exists(ram_fn): os.remove(ram_fn)
fd = io.open(ram_fn, 'w' )
fd.write('step,comment,ram\n')
fd.close()


os_info = os.uname()
global num_cores
if os_info[1] == 'imenb079':
    num_cores = 1 
else :
    num_cores = min(14, multiprocessing.cpu_count() )

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
  
def RAM(comment='') : 
    global ram_counter
    fd = io.open(ram_fn, 'a')
    ram = psutil.virtual_memory()[2]
    ram_string=f'{ram_counter},{comment},{ram}\n'
    fd.write(ram_string)
    ram_counter += 1
    fd.close()
    print(ram_string,end='')

@timer_func
def display_top(snapshot, key_type='lineno', limit=20):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)
    unit = 1073741824
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("\t#%s: %s:%s: %3.2f GiB"
              % (index, frame.filename, frame.lineno, float(stat.size) / unit))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("\t%s other: %.2f GiB" % (len(other), size / unit))
    total = sum(stat.size for stat in top_stats)
    print("\tTotal allocated size: %.2f GiB" % (total / unit))


def print_size(obj): print(getsizeof(obj)*1e-9,type(obj))

def plot_faces(coords, faces, out_fn):
    patches=[]
    fig, ax = plt.subplots(1,1)
    for a,b,c in faces:
        polygon = Polygon( [coords[a][0:2], coords[b][0:2], coords[c][0:2]], True, ec='b',lw=3)
        patches.append(polygon)

    colors = 10 * np.random.rand(len(patches))
    p = PatchCollection(patches, alpha=0.7)
    p.set_array(colors)
    ax.add_collection(p)
    ax.set_xlim(0,20)
    ax.set_ylim(0,20)
    plt.scatter(coords[:,0],coords[:,1])
    for i in range(coords.shape[0]) :
        ax.annotate(str(i), (coords[i,0], coords[i,1]))
    plt.savefig(out_fn)


#def add_entry(d,a,lst):
#    try :
#        d[a] += lst 
#    except KeyError :
#        d[a] = lst
#    return d

def add_entry(d,a,lst, nngh):
    n = len(lst)
    nngh += n
    
    if nngh > d[a].shape[0] :
        #d[a].resize((nngh*2,))
        #new_d = np.zeros([d.shape[0],d.shape[1]+2))
        d = np.concatenate([d,np.zeros([d.shape[0],2])],axis=1)
        #print('\t\t\tResizing ngh',d.shape)
    d[a][nngh-n:nngh] = lst 

    return d, nngh


def get_ngh_h5(faces_h5_fn, ngh_h5_fn, nngh_npz_fn):

    triangles_h5_file = h5py.File(faces_h5_fn, 'r')
    triangles = triangles_h5_file['data'][:].astype(np.uint32)
    triangles_h5_file.close()
    max_idx=np.max(triangles)+1
    nngh=np.zeros([max_idx]).astype(np.uint32)

    #for a in range(max_idx):
    #    d.create_dataset(str(a), shape=(20,), maxshape=(None,), chunks=True, dtype=np.int32)
   
    d=np.zeros([max_idx,20])

    for count, (i,j,k) in enumerate(triangles):
        #if count % 1000 == 0: 
        #    print('\t\t%3.1f'%(count/triangles.shape[0]*100),end='\r')
        d, nngh[i] = add_entry(d,int(i),[int(j),int(k)], nngh[i])
        d, nngh[j] = add_entry(d,int(j),[int(i),int(k)], nngh[j])
        d, nngh[k] = add_entry(d,int(k),[int(j),int(i)], nngh[k])
    del triangles

    #def adjust_ngh( key):
    for key in range(d.shape[0]):
        unique_idx = np.array(np.unique(d[key][0:nngh[int(key)]]).astype(np.int32))
        d[key][:]=0
        d[key][0:unique_idx.shape[0]] = unique_idx
        nngh[int(key)] = len(unique_idx)
    
    #global num_cores 
    #Parallel(n_jobs=num_cores)(delayed(adjust_ngh)(key) for key in range(d.shape[0])) 
    
    np.savez(ngh_h5_fn, data=d)
    np.savez(nngh_npz_fn, data=nngh)

def save_gii(coords, triangles, reference_fn, out_fn):
    #print('ref fb', reference_fn)
    #print(coords[0:5])
    #print(triangles[0:5])
    assert len(coords) > 0, f'Empty coords when trying to create {out_fn} '
    assert len(triangles) > 0, f'Empty triangles when trying to create {out_fn} '
    img = load_mesh(reference_fn) 
    ar_pointset = nb.gifti.gifti.GiftiDataArray(data=coords.astype(np.float32), intent='NIFTI_INTENT_POINTSET') 
    ar_triangle = nb.gifti.gifti.GiftiDataArray(data=triangles.astype(np.int32), intent='NIFTI_INTENT_TRIANGLE') 
    darrays=[ar_pointset,ar_triangle]
    #darrays=[ar2,ar1] #DEBUG FIXME
    out = nb.gifti.GiftiImage(darrays=darrays, header=img.header, file_map=img.file_map, extra=img.extra, meta=img.meta, labeltable=img.labeltable) 
    out.to_filename(out_fn) 
    #print(out.print_summary())

def obj_to_gii(in_fn, gii_ref_fn, gii_out_fn):
    if '.obj' in in_fn :
        coords, faces = read_obj(obj_fn)
    elif '.pial' in in_fn or '.white' in in_fn :
        coords, faces = freesurfer.io.read_geometry(in_fn)
    else :
        print('Error: filetype not supported for', in_fn)
        exit(1)

    save_gii(coords, faces, gii_ref_fn, gii_out_fn)

def calc_dist (x,y) : return np.sqrt(np.sum(np.power((x - y),2), axis=1))

def find_opposing_polygons(faces):
    opposing_faces=np.zeros(faces.shape[0]).astype(int)
    for i, (a, b, c) in enumerate(faces) :
        ar = np.intersect1d(ngh[a], ngh[b]).astype(int)
        ar = ar[ ar != c ]
        d = ar[0] 
        temp_poly = sorted([a,b,d])
        opposing_faces[i] = faces[ (faces[:,0] == temp_poly[0]) & (faces[:,1] == temp_poly[1]) & (faces[:,2] == temp_poly[2]) ]
        
    return opposing_faces


def sorted_str (l): 
    sorted_l=sorted(l)
    temp_list = [ str(element) for element in sorted_l ]
    output_string = ''.join(temp_list)
    return output_string

def calculate_new_coords(coords, f_i, f_j, idx):
    c_i = coords[f_i][idx] 
    c_j = coords[f_j][idx]

    new_coords = ( c_i + c_j )/2.
    
    #for f0,f1, i, j, k in zip(f_i, f_j, c_i, c_j, new_coords):print(f'{f0} {f1}\t{i} + {j} = {k}')
    new_coords = np.concatenate([coords, new_coords])

    return new_coords

import pandas as pd


def calc_new_coords(faces_h5_fn, coords_h5_fn, resolution, n_new_edges, new_edges_npz_fn, surf_slab_space_dict=None):
    # Open h5py file
    faces_h5 = h5py.File(faces_h5_fn, 'r')
    coords_h5 = h5py.File(coords_h5_fn, 'a')

    coord_offset = coords_h5['data'].shape[0]
    faces_offset = faces_h5['data'].shape[0] 

    #get the index number of all the edges in the mesh
    edges = get_edges_from_faces(faces_h5['data']) 
    if type(surf_slab_space_dict) == dict :
        edge_mask = identify_target_edges(edges, surf_slab_space_dict, new_edges_npz_fn)
        #remove edges that are not in the mask, i.e., edges which do not intersect a histological section
        edges = edges[edge_mask]

    e_0, e_1 = edges[:,0], edges[:,1]

    n_edges = e_0.shape[0]
    
    c_0 = coords_h5['data'][:][e_0] 
    c_1 = coords_h5['data'][:][e_1]

    #calculate the euclidean distance between all points on an edge
    d_ij = calc_dist(c_0, c_1)

    max_edge_length = np.max(d_ij)
    mean_edge_length = np.mean(d_ij)
    perc95 = np.percentile(d_ij,[95])[0]
    mesh_stats={'max':max_edge_length, 'mean':mean_edge_length, 'perc95':perc95}

    long_edges = d_ij >= resolution
   
    n_valid=np.sum(~long_edges)
    n_long=np.sum(long_edges).astype(int)

    n_total_coords = coord_offset + n_long

    coords_h5['data'].resize((n_total_coords,3))
    coords_h5['data'][coord_offset:] = ( c_0[long_edges] + c_1[long_edges] )/2.
    
    coords_h5.close()
    faces_h5.close()
    
    n_total_new_edges = n_new_edges+ n_long

    del e_0
    del e_1
    del d_ij
    del c_0
    del c_1
    return edges, n_total_new_edges, long_edges, n_long, coord_offset, faces_offset, n_edges, mesh_stats


def get_opposite_poly(edges, edge_counter, ngh, nngh, faces_dict, debug=False ): 
    a = int(edges[edge_counter][0])
    b = int(edges[edge_counter][1])
    #ab_ngh = list_intersect(ngh[a],ngh[b]) 
    #print('nngh', nngh[int(a)])
    #print( ngh[a][0:nngh[int(a)]], ngh[b][0:nngh[int(b)]] )
    ab_ngh = list_intersect(list(ngh[a][0:nngh[int(a)]]), list(ngh[b][0:nngh[int(b)]]) ) 
    #print('ab_ngh', ab_ngh)
    if debug :
        assert len(ab_ngh) == 2 , 'more than two neighbours for vertices {} {}'.format(a,b)

    c = int(ab_ngh[0])
    d = int(ab_ngh[1])
    del ab_ngh
    #print('a, b', a,b)
    #print('c, d', c,d)

    #if not a in ngh : 
    #    print('a not in ngh')
    #    exit(0)

    #if not b in ngh : 
    #    print('b not in ngh')
    #    exit(0)
    #if not c in ngh : 
    #    print('c not in ngh')
    #    exit(0)
    #if not d in ngh : 
    #    print('d not in ngh')
    #    exit(0)
    #print(a,b,c)
    #print(sorted([int(a),int(b),int(c)]))
    abc=[int(a),int(b),int(c)]
    face_idx=faces_dict[sorted_str(abc)]
    del abc

    #print(a,b,d)
    #ar = list_intersect(ngh[a], ngh[b])
    abd = sorted_str([int(a),int(b),int(d)])
    opposite_poly_index = faces_dict[abd]
    del abd
    #the idea is that the new coordinate that was interpolated between vertex a and b is stored
    #in new_coords[index]
    if debug :
        assert len(list_intersect(ngh[a],ngh[b])) == 2, 'a b index'
        assert len(list_intersect(ngh[a],ngh[c])) == 2, 'a c index'
        assert len(list_intersect(ngh[a],ngh[d])) == 2, 'a d index'
    return face_idx, opposite_poly_index, a, b, c, d

def update_faces(ngh, nngh, faces_h5, faces_dict, face_idx, faces_offset, index, opposite_poly_index, a, b, c, d, debug=False ):
    #insert new vertex in mesh --> 4 new polygons
    #new_faces[face_idx] = sorted([a,index,c]) 
    #new_faces[opposite_poly_index] = sorted([index,c,b])
    #new_faces[faces_offset] = sorted([a,index,d]) 
    #new_faces[faces_offset+1] = sorted([index,b,d])

    faces_h5[face_idx] = sorted([a,index,c]) 
    faces_h5[opposite_poly_index] = sorted([index,c,b])
    faces_h5[faces_offset] = sorted([a,index,d]) 
    faces_h5[faces_offset+1] = sorted([index,b,d])
    
    #faces_h5['data'][face_idx] = sorted([a,index,c]) 
    #faces_h5['data'][opposite_poly_index] = sorted([index,c,b])
    #faces_h5['data'][faces_offset] = sorted([a,index,d]) 
    #faces_h5['data'][faces_offset+1] = sorted([index,b,d])

    aic=[a,index,c]
    icb=[index,c,b]
    aid=[a,index,d]
    ibd=[index,b,d]
    faces_dict[sorted_str(aic)] = face_idx
    faces_dict[sorted_str(icb)] = opposite_poly_index
    faces_dict[sorted_str(aid)] = faces_offset
    faces_dict[sorted_str(ibd)] = faces_offset+1
    del aic
    del icb
    del aid
    del ibd

    #update the neighbours of each vertex to reflect changes to mesh
    #print('old_a_ngh', ngh[str(a)][:])
    #print('old_b_ngh', ngh[str(b)][:])
    new_a_ngh = [ int(ii) if int(ii) != int(b) else index for ii in ngh[int(a)][0:nngh[int(a)]] ]
    new_b_ngh = [ int(ii) if int(ii) != int(a) else index for ii in ngh[int(b)][0:nngh[int(b)]] ]
    #print('new_a_ngh', new_a_ngh)
    #print('new_b_ngh', new_b_ngh)
    nngh[a] = len(new_a_ngh)
    nngh[b] = len(new_b_ngh)
    ngh[int(a)][0:nngh[a]] = new_a_ngh 
    ngh[int(b)][0:nngh[b]] = new_b_ngh #[ ii if ii != a else index for ii in ngh[b] ]

    if nngh[c] >= ngh[int(c)].shape[0] : ngh[int(c)].resize(nngh[c]*2)
    if nngh[d] >= ngh[int(d)].shape[0] : ngh[int(d)].resize(nngh[d]*2)

    #print(ngh[int(c)][:])
    #print(ngh[int(d)][:])
    ngh[int(c)][nngh[c]] = index
    ngh[int(d)][nngh[d]] = index

    nngh[int(c)] += 1 
    nngh[int(d)] += 1 

    #print('new index', index)
    #print(ngh[str(c)][:])
    #print(ngh[str(d)][:])
    #ngh[c].append(index) 
    #ngh[d].append(index) 

    #ngh.create_dataset(str(index), (10,), dtype=np.uint32)
    nngh[index] = 4
    ngh[int(index)][0:4] = [a,b,c,d]

    if debug :
        assert len(list_intersect(ngh[a],ngh[c])) == 2, 'a c'
        assert len(list_intersect(ngh[c],ngh[b])) == 2, 'c b {} {} {}'.format(ngh[c],ngh[b], list_intersect(ngh[c],ngh[b]))
        assert len(list_intersect(ngh[a],ngh[d])) == 2, 'a d'
        assert len(list_intersect(ngh[b],ngh[d])) == 2, 'b d'
        assert len(list_intersect(ngh[a],ngh[index])) == 2, 'a index'
        assert len(list_intersect(ngh[b],ngh[index])) == 2, 'b index'
        assert len(list_intersect(ngh[c],ngh[index])) == 2, 'c index'
        assert len(list_intersect(ngh[d],ngh[index])) == 2, 'd index'

    return faces_dict

#def list_intersect( x, y) : return list(set([ii for ii in x+y if ii in x and ii in y]))
def list_intersect( x, y) : return list(set([ii for ii in x+y if ii in x and ii in y]))

def assign_new_edges(new_edges_npz_fn, edges,  n_new_edges, n_total_new_edges, long_edges):
    # new_edges are the edgges that define the upsampled mesh

    long_edges=np.arange(long_edges.shape[0])[long_edges].astype(np.uint32)
    
    temp_fn=new_edges_npz_fn+'_temp.h5'
    edges_temp = h5.File(temp_fn,'w')

    edges_temp.create_dataset('data',edges.shape,dtype=np.uint32)

    edges_temp['data'][:] = edges
    edges_temp.close()

    temp_edges = np.vstack([edges[long_edges,0],edges[long_edges,1]])
    del edges
    
    temp_edges = temp_edges.T
    
    if n_new_edges == 0 :
        new_edges = temp_edges
    else :
        new_edges = np.load(new_edges_npz_fn+'.npz', 'a')['data']
        new_edges = np.concatenate([new_edges, temp_edges], axis=0)

    np.savez(new_edges_npz_fn, data=new_edges)

    return temp_fn


def subdivide_triangle(coord_offset, coord_idx, edges, edge_counter, ngh, nngh, faces_dict, faces_h5, faces_offset):
    index = coord_offset + coord_idx

    face_idx, opposite_poly_index, a, b, c, d = get_opposite_poly(edges['data'], edge_counter, ngh, nngh, faces_dict )

    faces_dict = update_faces(ngh, nngh, faces_h5, faces_dict, face_idx, faces_offset, index, opposite_poly_index, int(a), int(b), int(c), int(d) )

    del face_idx
    del opposite_poly_index
    del a
    del b
    del c 
    del d

def upsample_edges(output_dir, coords_h5_fn, faces_h5_fn, faces_dict, new_edges_npz_fn,  resolution, surf_slab_space_dict=None, temp_alt_coords=None, debug=False, n_new_edges=0, coord_normals = []) :

    edges, n_total_new_edges, long_edges, n_long, coord_offset, faces_offset, n_edges, mesh_stats = calc_new_coords(faces_h5_fn, coords_h5_fn, resolution, n_new_edges, new_edges_npz_fn, surf_slab_space_dict=surf_slab_space_dict)
    n_coords = coord_offset +  n_long

    temp_fn = assign_new_edges(new_edges_npz_fn, edges, n_new_edges, n_total_new_edges, long_edges)
    edges=h5.File(temp_fn, 'r')
    
    ngh_h5_fn = f'{output_dir}/ngh_{resolution}_rsl_{n_total_new_edges}'
    nngh_npz_fn = f'{output_dir}/nngh_{resolution}_rsl_{n_total_new_edges}'
    if not os.path.exists(ngh_h5_fn) or not os.path.exists(nngh_npz_fn+'.npz') :
        get_ngh_h5(faces_h5_fn, ngh_h5_fn, nngh_npz_fn)
     
    ngh = np.load(ngh_h5_fn+'.npz')['data'].astype(np.uint32)
    nngh = np.load(nngh_npz_fn+'.npz')['data'].astype(np.uint32)
    #extend nngh for the total number of new coordinates
    nngh = np.concatenate([nngh,-1*np.ones(n_long)],axis=0).astype(np.uint32) 
    ngh = np.concatenate([ngh,np.ones([n_total_new_edges,ngh.shape[1]])],axis=0).astype(np.uint32) 
    
    faces_h5_file = h5py.File(faces_h5_fn, 'r')
    faces_h5 = faces_h5_file['data'][:]
    faces_h5_file.close()
    n_new_faces = faces_h5.shape[0]+n_long*2

    faces_h5 = np.concatenate([faces_h5,-1*np.ones([n_long*2,3])],axis=0).astype(np.uint32) 
    
    #
    #           d
    #          /|\
    #         / | \ 
    #        /  |  \
    #       /   |   \
    #     a/____|____\ b
    #      \   I|    /
    #       \   |   /
    #        \  |  /
    #         \ | /
    #          \|/
    #           c
    #
    #   a,I,B ; b,c,I ; a, d, I ; d, b, I
    edges_range = enumerate(np.arange(n_edges).astype(np.uint32)[long_edges])
    del long_edges

    #coords_h5 = h5.File(coords_h5_fn)
    #coords = coords_h5['data']

    #iterate over long edges (i.e., greater than desired resolution)
    for coord_idx, edge_counter in edges_range:

        #if coord_idx < 20 :
        #    print(coord_idx, edges['data'][edge_counter])

        if coord_idx % 50000 ==0 : print('\t\t{}'.format(np.round(100*coord_idx/n_long,2))) #,end='\r')

        #NOTE: faces_dict is a dictionary that maps the index of a face on the surface mesh
        # to an index inthe faces_h5 numpy array. This allows the main data to be stored in the
        # numpy array while accessing it through a dictionary
        subdivide_triangle(coord_offset, coord_idx, edges, edge_counter, ngh, nngh, faces_dict, faces_h5, faces_offset)
        # add two new faces to total number of faces (we go from 2 faces to 4, so net gain of 2) 
        faces_offset += 2

    del edges
    del ngh

    faces_h5_file = h5py.File(faces_h5_fn, 'w')
    faces_h5_file.create_dataset('data', faces_h5.shape, dtype=np.uint32)
    faces_h5_file['data'][:] = faces_h5
    faces_h5_file.close()
    del faces_h5
    
    os.remove(temp_fn)

    #max_len, avg_len, perc95, n_coords, n_faces = get_mesh_stats(faces_h5_fn, coords_h5_fn, edge_mask=edge_mask)
    print(f'\tmax edge {mesh_stats["max"]:.3f}\tavg: {mesh_stats["mean"]:.3f}\tperc95: {mesh_stats["perc95"]:.3f}\tcoords= {n_coords}' )
    new_coord_normals = [] 
    return  mesh_stats['max'], new_coord_normals, faces_dict, n_total_new_edges



def setup_coordinate_normals(faces,normals,coords):
    new_normals=np.zeros(faces.shape)
    coord_normals = np.zeros_like(coords)
    for (a, b, c), n in zip(faces[0:normals.shape[0]], normals) :
        coord_normals[a] += n
        coord_normals[b] += n
        coord_normals[c] += n
    return coord_normals

def fix_normals(faces,coords,coord_normals):
    for i in range(faces.shape[0]) :
        a,b,c = faces[i]
        test_normal = np.cross(coords[b]-coords[a], coords[c]-coords[a])
        average_normal=coord_normals[a]+coord_normals[b]+coord_normals[c]
        x=np.dot(average_normal, test_normal)
        if x < 0 : faces[i]=[c,b,a]
    return faces

#def write_mesh( coords, faces, input_fn, upsample_fn ):
#    ext = upsample_fn.split('.')[-1]
#    if ext == 'gii' :
#        save_gii( coords, faces, input_fn, upsample_fn )
#    elif ext == 'obj' :
#        save_obj(upsample_fn,coords,faces)
#    else :
#        print('not implemented for ext', ext)
#        exit(1)

def write_gifti_from_h5(upsample_fn, coords_fn, faces_fn, input_fn ) :
    print('\tFrom coords:', coords_fn)
    print('\tand faces:',faces_fn)
    print('\tWriting', upsample_fn)

    coords_h5 = h5py.File(coords_fn,'r')
    faces_h5 = h5py.File(faces_fn,'r')
    
    if '.pial' in input_fn or '.white' in input_fn :
        volume_info = load_mesh(input_fn)[2]
    else :
        volume_info = input_fn

    save_mesh(upsample_fn, coords_h5['data'][:], faces_h5['data'][:], volume_info)
    #if temp_alt_coords != None :
    #    for orig_fn, fn in temp_alt_coords.items() :
    #        alt_upsample_fn=sub('.surf.gii','_rsl.surf.gii',orig_fn)
    #        print('\tWriting',alt_upsample_fn)
    #        write_mesh(np.load(fn+'.npy'), faces, input_fn, alt_upsample_fn )


def setup_h5_arrays(input_fn, upsample_fn, faces_h5_fn, coords_h5_fn,  clobber=False):
    ext = os.path.splitext(input_fn)[1]
    
    coords_npy, faces_npy, volume_info = load_mesh(input_fn,correct_offset=False)
    
    faces_h5 = h5py.File(faces_h5_fn,'w')
    faces_h5.create_dataset('data', data=faces_npy, maxshape=(None, 3), dtype=np.int32)
    del faces_npy

    coords_h5 = h5py.File(coords_h5_fn,'w')
    coords_h5.create_dataset('data', data=coords_npy, maxshape=(None, 3), dtype=np.float16)
    del coords_npy


    faces_h5['data'][:] = np.sort(faces_h5['data'][:], axis=1).astype(np.int32)
    faces_dict={}
    for i, (x,y,z) in enumerate(faces_h5['data'][:]):
        xyz=[x,y,z]
        key=sorted_str([x,y,z])
        del xyz
        faces_dict[key] = i

    return faces_dict


def get_mesh_stats(faces_h5_fn, coords_h5_fn, edge_mask=None):
    faces_h5 = h5py.File(faces_h5_fn,'r')
    coords_h5 = h5py.File(coords_h5_fn,'r')

    edges = get_edges_from_faces(faces_h5['data'][:])

    e0=edges[:,0]
    e1=edges[:,1]

    #calculate edge lengths
    d = calc_dist(coords_h5['data'][:][ e0 ], coords_h5['data'][:][e1])

    max_len = np.max(d)
    avg_len = np.round(np.mean(d),3)
    perc95 = np.round(np.percentile(d,[95])[0],3)
    n_coords =  coords_h5['data'][:].shape[0]
    n_faces = faces_h5['data'][:].shape[0] 
    
    coords_h5.close()
    faces_h5.close()

    del d

    return max_len, avg_len, perc95, n_coords, n_faces


def upsample_with_h5(input_fn,upsample_fn, faces_h5_fn, coords_h5_fn, new_edges_npz_fn, resolution, surf_slab_space_dict=None, test=False, clobber=False, debug=False):
        output_dir = os.path.dirname(upsample_fn)
        print('setup h5 arrays')
        faces_dict = setup_h5_arrays(input_fn, upsample_fn, faces_h5_fn, coords_h5_fn, clobber=clobber)
        #calculate surface normals
        coord_normals=[]
        #if resolution > 1 :
        #    faces_h5 = h5.File(faces_h5_fn, 'r')
        #    coords_h5 = h5.File(coords_h5_fn, 'r')
        #    normals = np.array([ np.cross(coords_h5['data'][b]-coords_h5['data'][a],coords_h5['data'][c]-coords_h5['data'][a]) for a,b,c in faces_h5['data'][:] ])
        #    coord_normals = setup_coordinate_normals(faces_h5['data'][:],normals,coords_h5['data'][:])
        #    del faces_h5
        #    del coords_h5

        print('get mesh stats')
        max_len, avg_len, perc95, n_coords, n_faces = get_mesh_stats(faces_h5_fn, coords_h5_fn)
        print('\tmax edge',max_len,'\tavg', avg_len,'\tcoords=', n_coords , '\tfaces=',n_faces )

        counter=0
        n_new_edges = 0
        metric = max_len

        #print('target reslution:', resolution)
        while metric > resolution :
            old_metric = metric
            print('Number of too long vertices', metric)
            metric, coord_normals, faces_dict, n_new_edges = upsample_edges(output_dir, coords_h5_fn, faces_h5_fn, faces_dict, new_edges_npz_fn, resolution, surf_slab_space_dict=surf_slab_space_dict, debug=debug,  n_new_edges=n_new_edges, coord_normals=coord_normals)
            counter+=1
        if coord_normals != [] :
            faces=fix_normals(faces,coords_h5['data'][:],coord_normals)

        del faces_dict 

        return n_new_edges

def identify_target_edges_within_slab(edge_mask, ligand_vol_fn, coords, edges, ext='.nii.gz'):
    
    img = nib.load(ligand_vol_fn)
    ligand_vol = img.get_fdata()
    step = img.affine[1,1]
    start = img.affine[1,3]

    e0 = edges[:,0]
    e1 = edges[:,1]
    c0 = coords[e0,1]
    c1 = coords[e1,1]
   
    starting_edge = np.argmax(edge_mask)

    edge_y_coords = np.vstack([c0,c1]).T

    idx_1 = np.argsort(edge_y_coords,axis=1)
    edges = np.take_along_axis(edge_y_coords, idx_1, 1)
    sorted_edge_y_coords = np.take_along_axis(edge_y_coords, idx_1, 1)
    
    idx_0 = np.argsort(edges,axis=0)
    sorted_edge_y_coords = np.take_along_axis( sorted_edge_y_coords, idx_0, 0)
    edges = np.take_along_axis( edges, idx_0, 0)

    ligand_y_profile = np.sum(ligand_vol,axis=(0,2))
    section_numbers = np.where(ligand_y_profile > 0)[0]
    
    section_counter=0
    current_section_vox = section_numbers[section_counter]
    current_section_world = current_section_vox * step + start

    for i in np.arange(starting_edge, edges.shape[0]).astype(int) :
        y0,y1 = sorted_edge_y_coords[i,:]
        e0,e1 = edges[i]
        crossing_edge = (y0 < current_section_world) & (y1 > current_section_world + step)
        start_in_edge = ((y0 >= current_section_world) & (y0 < current_section_world+step)) & (y1 > current_section_world + step)
        end_in_edge = (y0 < current_section_world) & ((y1>current_section_world) & (y1 <= current_section_world + step))
        
        if crossing_edge + start_in_edge + end_in_edge > 0 :
            #print('y0-y1', y0, y1, 'sec',current_section_world)
            edge_mask[i]=True

        if y0 > current_section_world :
            section_counter += 1
            if section_counter >= section_numbers.shape[0] :
                break
            current_section_vox = section_numbers[section_counter]
            current_section_world = current_section_vox * step + start
    
    return edge_mask 


def identify_target_edges(edges, slab_dict, new_edges_npz_fn, ext='.surf.gii'):
    
    edge_mask = np.zeros(edges.shape[0]).astype(np.bool)

    slab_list = list(slab_dict.keys())
    slab_list.sort()

    if os.path.exists(new_edges_npz_fn+'.npz'):
        # load new edges for resampling of slab space mesh
        new_edges = np.load(new_edges_npz_fn+'.npz')['data'][:]

    #print(slab_list)
    for slab in slab_list :
        cdict = slab_dict[slab]
        #NOTE should actually recalculate this at every iteration because as the edges get smaller
        # there are less of them that will cross acquire sections617gg
        coords, _, _ = load_mesh(cdict['surf'])
        n_coords = coords.shape[0]
        
        if os.path.exists(new_edges_npz_fn+'.npz'): 
            #if this is not hte first iteration of the upsampling, then we need to 
            #add new upsampled points based on the new edges
            coords = np.concatenate([coords,np.zeros([new_edges.shape[0]+1,3])],axis=0)
            coords = generate_upsampled_coordinates(new_edges, coords, n_coords)
            
        edge_mask = identify_target_edges_within_slab(edge_mask, cdict['vol'], coords, edges)
        
        print(f'\tSlab {slab}: % of edges to be split:', 100.0*np.sum(edge_mask)/edge_mask.shape[0])

    return edge_mask

def generate_upsampled_coordinates(new_edges, rsl_coords, n_coords):
    n_edges=new_edges.shape[0]
    for ii in range(n_edges) :
        e0, e1 = new_edges[ii,:]
        rsl_coords[ n_coords + ii, : ] = (rsl_coords[int(e0),:] + rsl_coords[int(e1),:])/2.
    return rsl_coords

def resample_gifti_to_h5(new_edges_npz_fn, reference_coords_h5_fn, input_list, output_list) :

    new_edges_h5 = np.load(new_edges_npz_fn+'.npz')['data']
    reference_coords_h5 = h5py.File(reference_coords_h5_fn, 'r')
    n = reference_coords_h5['data'].shape[0]
    n_edges = new_edges_h5.shape[0]

    for i, (in_fn, out_fn) in enumerate(zip(input_list,output_list)):
        if not os.path.exists(out_fn) :
            coords = load_mesh(in_fn, correct_offset=False)[0] 
            n_coords = coords.shape[0]
            assert n == n_coords + n_edges, f'Error: number of resampled coordinates ({n}) does not equal the number of original coordiantes ({n_coords}) + number of new coordinates ({n_edges}).'
            rsl_coords_h5 = h5py.File(out_fn, 'w')
            rsl_coords_h5.create_dataset('data', (n,3) )
            rsl_coords_h5['data'][ 0 : n_coords] = coords
            rsl_coords = rsl_coords_h5['data'][:]

            new_edges = new_edges_h5[:].astype(int)
            if n_edges > 0 :
                rsl_coords = generate_upsampled_coordinates(new_edges, rsl_coords, n_coords)
            else : 
                print('No extra coordinates to interpolate')

            if np.sum(np.sum( np.abs(rsl_coords), axis=1) == 0) > 1 : 
                print("Error: got multiple 0,0,0 vertices after resampling")
                os.remove(out_fn)
                exit(1)

            rsl_coords_h5['data'][:] = rsl_coords
            rsl_coords_h5.close()
            print('Closing', out_fn)


def upsample_gifti(input_fn,upsample_0_fn, upsample_1_fn, resolution, input_list=[], output_list=[], surf_slab_space_dict=None, test=False, clobber=False, debug=False):
    tracemalloc.start()

    if '.surf.gii' in upsample_0_fn : ext = '.surf.gii'
    elif '.white' in upsample_0_fn : ext = '.white'
    elif '.pial' in upsample_0_fn : ext = '.pial'
    elif '.obj' in upsample_0_fn : ext = '.obj'
    else :
        print('Error: no extension found for', upsmaple_0_fn)
        exit(1)

    assert os.path.exists(input_fn) , 'Error, missing '+ input_fn

    faces_h5_fn =  sub(ext,'_new_faces.h5',upsample_0_fn)
    coords_h5_fn =   sub(ext,'_new_coords.h5',upsample_0_fn)
    new_edges_npz_fn =   sub(ext,'_new_edges',upsample_0_fn)

    if not os.path.exists(coords_h5_fn) or not os.path.exists(new_edges_npz_fn+'.npz') :
        n_new_edges = upsample_with_h5(input_fn, upsample_0_fn,  faces_h5_fn, coords_h5_fn, new_edges_npz_fn, resolution, surf_slab_space_dict=surf_slab_space_dict)

    #if not os.path.exists(upsample_0_fn) :
        write_gifti_from_h5(upsample_0_fn, coords_h5_fn, faces_h5_fn, input_fn ) 
    
    if input_list != []  and output_list != [] :
        resample_gifti_to_h5(new_edges_npz_fn, coords_h5_fn, input_list, output_list)

    if not os.path.exists(upsample_1_fn) :
        write_gifti_from_h5(upsample_1_fn, output_list[-2], faces_h5_fn, input_fn ) 

    return faces_h5_fn, coords_h5_fn

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', dest='input_fn',  default=None, help='input fn')
    parser.add_argument('-r', dest='resolution',  default=None, help='resolution')
    parser.add_argument('-a', dest='input_list', nargs='+', default=[], help='optional fn')
    parser.add_argument('-c', action='store_true', default=False, dest='clobber', help='optional fn')
    parser.add_argument('-t', action='store_true', default=False, dest='test', help='test upsampling')
    parser.add_argument('-d', action='store_true', default=False, dest='debug', help='debug')
    args = parser.parse_args()
    
    upsample_fn=''
    if not args.test:
        basename, ext = splitext(args.input_fn)
        upsample_fn = f'{basename}_rsl{ext}' 
    if not os.path.exists(upsample_fn) or args.clobber or args.test  or True:
        upsample_gifti( args.input_fn, upsample_fn, float(args.resolution), input_list=args.input_list,  clobber=args.clobber, test=args.test, debug=args.debug)

