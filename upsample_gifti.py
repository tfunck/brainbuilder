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
from guppy import hpy
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from utils.mesh_io import load_mesh_geometry, save_obj, read_obj
from re import sub
from utils.utils import shell,splitext
from glob import glob
from sys import getsizeof

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


def add_entry(d,a,lst):
    try :
        #d[a]=np.append(d[a], lst) 
        d[a] += lst 
    except KeyError :
        d[a] = lst
    return d

def get_ngh(triangles):
    d={}
    for i,j,k in triangles:
        d = add_entry(d,i,[j,k])
        d = add_entry(d,j,[i,k])
        d = add_entry(d,k,[j,i])

    for key in d.keys():
        #d[key] =np.unique(d[key])
        d[key] = list(np.array(np.unique(d[key])))
    del triangles 
    return d 

def save_gii(coords, triangles, reference_fn, out_fn):
    img = nb.load(reference_fn) 
    ar1 = nb.gifti.gifti.GiftiDataArray(data=coords.astype(np.float32), intent='NIFTI_INTENT_POINTSET') 
    ar2 = nb.gifti.gifti.GiftiDataArray(data=triangles.astype(np.int32), intent='NIFTI_INTENT_TRIANGLE') 
    out = nb.gifti.GiftiImage(darrays=[ar1,ar2], header=img.header, file_map=img.file_map, extra=img.extra, meta=img.meta, labeltable=img.labeltable) 
    out.to_filename(out_fn) 
    #print(out.print_summary())

def obj_to_gii(obj_fn, gii_ref_fn, gii_out_fn):
    coords, faces = read_obj(obj_fn)
    save_gii(coords, faces, gii_ref_fn, gii_out_fn)

def calc_dist (x,y) : return np.sum(np.abs(x - y), axis=1)

def find_opposing_polygons(faces):
    opposing_faces=np.zeros(faces.shape[0]).astype(int)
    for i, (a, b, c) in enumerate(faces) :
        ar = np.intersect1d(ngh[a], ngh[b]).astype(int)
        ar = ar[ ar != c ]
        d = ar[0] 
        temp_poly = sorted([a,b,d])
        opposing_faces[i] = faces[ (faces[:,0] == temp_poly[0]) & (faces[:,1] == temp_poly[1]) & (faces[:,2] == temp_poly[2]) ]
        
    return opposing_faces


def sorted_str (l): return ''.join([ str(element) for element in sorted(l) ])

def calculate_new_coords(coords, f_i, f_j, idx):
    c_i = coords[f_i][idx] 
    c_j = coords[f_j][idx]

    new_coords = ( c_i + c_j )/2.
    
    #for f0,f1, i, j, k in zip(f_i, f_j, c_i, c_j, new_coords):print(f'{f0} {f1}\t{i} + {j} = {k}')
    new_coords = np.concatenate([coords, new_coords])

    return new_coords

import pandas as pd

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
    edges_all = np.concatenate([f_ij,f_jk, f_ki],axis=0)

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
    sorted_indices = edges_range_sorted[:,2] % faces.shape[0]
    # the edges are reshuffled once by sorting them by row and then by extracting unique edges
    # we need to keep track of these transformations so that we can relate the shuffled edges to the 
    # faces they belong to.
    edges, edges_idx, counts = np.unique(edges_sorted , axis=0, return_index=True, return_counts=True)
    #print('2.', np.sum( np.sum(edges_all==(26, 22251),axis=1)==2 ) ) 
    
    assert np.sum(counts!=2) == 0,'Error: more than two faces per edge {}'.format( edges_sorted[edges_idx[counts!=2]])     
    #edge_range = np.arange(edges_all.shape[0]).astype(int) % faces.shape[0]
    return edges

def calc_new_coords(faces_h5_fn, coords_h5_fn, resolution, n_new_edges):
    # Open h5py file
    faces_h5 = h5py.File(faces_h5_fn, 'r')
    coords_h5 = h5py.File(coords_h5_fn, 'a')

    coord_offset = coords_h5['data'][:].shape[0]
    faces_offset = faces_h5['data'][:].shape[0] 

    edges = get_edges_from_faces(faces_h5['data'][:]) 
    e_0, e_1 = edges[:,0], edges[:,1]

    n_edges = e_0.shape[0]
    
    c_0 = coords_h5['data'][:][e_0] 
    c_1 = coords_h5['data'][:][e_1]


    d_ij = calc_dist(c_0, c_1)

    long_edges = d_ij >= resolution
    mm = np.argmax(long_edges)
    
    n_valid=np.sum(~long_edges)
    n_long=np.sum(long_edges).astype(int)

    n_total_coords = coord_offset + n_long

    #new_coord_normals=[]
    #if coord_normals  != [] :
    #    new_coord_normals = calculate_new_coords(coord_normals, e_0, e_1, long_edges)

    coords_h5['data'].resize((n_total_coords,3))
    coords_h5['data'][coord_offset:] = ( coords_h5['data'][:][e_0][long_edges] + coords_h5['data'][:][e_1][long_edges] )/2.
    coords_h5.close()
    faces_h5.close()
    
    n_total_new_edges = n_new_edges+ n_long

    print('\t edges:',n_new_edges,n_total_new_edges, n_total_coords) 
    del e_0
    del e_1
    del d_ij
    del c_0
    del c_1
    return edges, n_total_new_edges, long_edges, n_long, coord_offset, faces_offset, n_edges


def get_opposite_poly(edges, edge_counter, ngh, faces_dict, debug=False ): 
    a = edges[edge_counter][0]
    b = edges[edge_counter][1]
    ab_ngh = list_intersect(ngh[a],ngh[b]) 
    if debug :
        assert len(ab_ngh) == 2 , 'more than two neighbours for vertices {} {}'.format(a,b)

    c = ab_ngh[0]
    d = ab_ngh[1]
    face_idx=faces_dict[sorted_str([a,b,c])]

    #ar = list_intersect(ngh[a], ngh[b])
    opposite_poly_index=faces_dict[sorted_str([a,b,d])]

    #the idea is that the new coordinate that was interpolated between vertex a and b is stored
    #in new_coords[index]
    if debug :
        assert len(list_intersect(ngh[a],ngh[b])) == 2, 'a b index'
        assert len(list_intersect(ngh[a],ngh[c])) == 2, 'a c index'
        assert len(list_intersect(ngh[a],ngh[d])) == 2, 'a d index'
    return face_idx, opposite_poly_index, a, b, c, d

def update_faces(edges, edge_counter, ngh, faces_h5, faces_dict, face_idx, faces_offset, index, opposite_poly_index, a, b, c, d, debug=False ):
    #insert new vertex in mesh --> 4 new polygons
    #new_faces[face_idx] = sorted([a,index,c]) 
    #new_faces[opposite_poly_index] = sorted([index,c,b])
    #new_faces[faces_offset] = sorted([a,index,d]) 
    #new_faces[faces_offset+1] = sorted([index,b,d])

    faces_h5['data'][face_idx] = sorted([a,index,c]) 
    faces_h5['data'][opposite_poly_index] = sorted([index,c,b])
    faces_h5['data'][faces_offset] = sorted([a,index,d]) 
    faces_h5['data'][faces_offset+1] = sorted([index,b,d])

    faces_dict[sorted_str([a,index,c])] = face_idx
    faces_dict[sorted_str([index,c,b])] = opposite_poly_index
    faces_dict[sorted_str([a,index,d])] = faces_offset
    faces_dict[sorted_str([index,b,d])] = faces_offset+1
    
    #update the neighbours of each vertex to reflect changes to mesh
    ngh[a] = [ ii if ii != b else index for ii in ngh[a] ] 
    ngh[b] = [ ii if ii != a else index for ii in ngh[b] ]
    ngh[c].append(index) 
    ngh[d].append(index) 

    ngh[index] = [a,b,c,d]

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

def list_intersect( x, y) : return list(set([ii for ii in x+y if ii in x and ii in y]))

def get_ngh_from_h5(faces_h5_fn) :
    faces_h5 = h5py.File(faces_h5_fn, 'r')
    ngh = get_ngh(faces_h5['data'][:])
    faces_h5.close()
    return ngh

def upsample_edges(coords_h5_fn, faces_h5_fn, faces_dict, new_edges_h5_fn,  resolution, temp_alt_coords=None, debug=False, n_new_edges=0, coord_normals = []) :

    edges, n_total_new_edges, long_edges, n_long, coord_offset, faces_offset, n_edges = calc_new_coords(faces_h5_fn, coords_h5_fn, resolution, n_new_edges)
    new_edges_h5 = h5py.File(new_edges_h5_fn, 'a')
    new_edges_h5['data'].resize((n_total_new_edges,2))
    new_edges_h5['data'][n_new_edges:n_total_new_edges]=np.vstack([edges[long_edges,0],edges[long_edges,1]]).T
    new_edges_h5.close()

    ngh = get_ngh_from_h5(faces_h5_fn) 
    
    faces_h5 = h5py.File(faces_h5_fn, 'a')
    n_new_faces = faces_h5['data'].shape[0]+n_long*2
    faces_h5['data'].resize( (n_new_faces, 3) )
    faces_h5['data'][faces_offset:,:] = -1

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
    edges_range = enumerate(np.arange(n_edges).astype(int)[long_edges])
    del long_edges

    new_faces_range = np.arange(n_new_faces).astype(int)
    #iterate over long edges (i.e., greater than desired resolution)

    for coord_idx, edge_counter in edges_range:
        if coord_idx % 10000 ==0 : 
            print('\t\t{}'.format(np.round(100*coord_idx/n_long,2))) #,end='\r')
            #print(h.heap(), h.heap().bysize[-1])
            #a = h.heap()
            #for i in range(int(len(a)/10)) :
            #    print(a)
            #    a = a.more

            #snapshot = tracemalloc.take_snapshot()
            #for stat_i, stat in enumerate(snapshot.statistics("lineno")): 
            #    print(stat)
            #    if stat_i > 10 : break

        index = coord_offset + coord_idx
        face_idx, opposite_poly_index, a, b, c, d = get_opposite_poly(edges, edge_counter, ngh, faces_dict )

        faces_dict = update_faces(edges, edge_counter, ngh, faces_h5, faces_dict, face_idx, faces_offset, index, opposite_poly_index, a, b, c,d )

        # add two new faces to total number of faces (we go from 2 faces to 4, so net gain of 2) 
        faces_offset += 2

        #get_edges_from_faces(new_faces[new_faces[:,0] != -1]) 
    del edges
    del ngh

    faces_h5.close() 

    max_len, avg_len, perc95, n_coords, n_faces = get_mesh_stats(faces_h5_fn, coords_h5_fn)
    print('\tmax edge',max_len,'\tavg', avg_len,'\tperc95', perc95, '\tcoords=', n_coords, '\tfaces=', n_faces )
    new_coord_normals = [] 
    return  max_len, new_coord_normals, faces_dict, n_total_new_edges



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

def write_mesh( coords, faces, input_fn, upsample_fn ):
    ext = upsample_fn.split('.')[-1]
    if ext == 'gii' :
        save_gii( coords, faces, input_fn, upsample_fn )
    elif ext == 'obj' :
        save_obj(upsample_fn,coords,faces)
    else :
        print('not implemented for ext', ext)
        exit(1)


def write_gifti_from_h5(upsample_fn, coords_fn, faces_fn, input_fn ) :
    print('\tFrom coords:', coords_fn)
    print('\tand faces:',faces_fn)
    print('\tWriting',upsample_fn)

    
    coords_h5 = h5py.File(coords_fn,'r')
    faces_h5 = h5py.File(faces_fn,'r')
    write_mesh( coords_h5['data'][:], faces_h5['data'][:], input_fn, upsample_fn )
    #if temp_alt_coords != None :
    #    for orig_fn, fn in temp_alt_coords.items() :
    #        alt_upsample_fn=sub('.surf.gii','_rsl.surf.gii',orig_fn)
    #        print('\tWriting',alt_upsample_fn)
    #        write_mesh(np.load(fn+'.npy'), faces, input_fn, alt_upsample_fn )


def setup_h5_arrays(input_fn, upsample_fn, faces_h5_fn, coords_h5_fn, new_edges_h5_fn, clobber=False):
    ext = os.path.splitext(input_fn)[1]
    if ext == '.gii' :
        mesh = nb.load(input_fn)
        faces_npy = mesh.agg_data('NIFTI_INTENT_TRIANGLE')
        coords_npy = mesh.agg_data('NIFTI_INTENT_POINTSET')
    elif ext == '.obj' :
        mesh_dict = load_mesh_geometry(input_fn)
        coords_npy = mesh_dict['coords']
        faces_npy = mesh_dict['faces']
    else :
        print('Error: could not recognize filetype from extension,',ext)
        exit(1)

    faces_h5 = h5py.File(faces_h5_fn,'w')
    faces_h5.create_dataset('data', data=faces_npy, maxshape=(None, 3))
    del faces_npy

    coords_h5 = h5py.File(coords_h5_fn,'w')
    coords_h5.create_dataset('data', data=coords_npy, maxshape=(None, 3))
    del coords_npy

    new_edges_h5 = h5py.File(new_edges_h5_fn,'w')
    new_edges_h5.create_dataset('data', (1,2), maxshape=(None, 2))
    new_edges_h5.close()

    faces_h5['data'][:] = np.sort(faces_h5['data'][:], axis=1).astype(np.int32)
    faces_dict={}
    for i, (x,y,z) in enumerate(faces_h5['data'][:]):
        key=sorted_str([x,y,z])
        faces_dict[key] = i

    return faces_dict


def get_mesh_stats(faces_h5_fn, coords_h5_fn):
    faces_h5 = h5py.File(faces_h5_fn,'r')
    coords_h5 = h5py.File(coords_h5_fn,'r')

    #calculate edge lengths
    d = [   calc_dist(coords_h5['data'][:][ faces_h5['data'][:,0] ], coords_h5['data'][:][ faces_h5['data'][:,1] ]),
            calc_dist(coords_h5['data'][:][ faces_h5['data'][:,1] ], coords_h5['data'][:][ faces_h5['data'][:,2] ]),
            calc_dist(coords_h5['data'][:][ faces_h5['data'][:,2] ], coords_h5['data'][:][ faces_h5['data'][:,0] ])]


    max_len = np.max(d)
    avg_len = np.round(np.mean(d),2)
    perc95 = np.round(np.percentile(d,[95])[0],3)
    n_coords =  coords_h5['data'][:].shape[0]
    n_faces = faces_h5['data'][:].shape[0] 
    
    coords_h5.close()
    faces_h5.close()

    del d

    return max_len, avg_len, perc95, n_coords, n_faces





def upsample_with_h5(input_fn,upsample_fn, faces_h5_fn, coords_h5_fn, new_edges_h5_fn, resolution, test=False, clobber=False, debug=False):
        
        faces_dict = setup_h5_arrays(input_fn, upsample_fn, faces_h5_fn, coords_h5_fn, new_edges_h5_fn, clobber=clobber)

        max_len, avg_len, perc95, n_coords, n_faces = get_mesh_stats(faces_h5_fn, coords_h5_fn)

        #calculate surface normals
        coord_normals=[]
        if resolution > 1 :
            faces_h5 = h5.File(faces_h5_fn, 'r')
            coords_h5 = h5.File(coords_h5_fn, 'r')
            normals = np.array([ np.cross(coords_h5['data'][b]-coords_h5['data'][a],coords_h5['data'][c]-coords_h5['data'][a]) for a,b,c in faces_h5['data'][:] ])
            coord_normals = setup_coordinate_normals(faces_h5['data'][:],normals,coords_h5['data'][:])
            del faces_h5
            del coords_h5
       
        print('\tmax edge',max_len,'\tavg', avg_len,'\tcoords=', n_coords , '\tfaces=',n_faces )

        counter=0
        n_new_edges = 0
        metric = max_len
        
        print('target reslution:', resolution)
        while metric > resolution :
            print(metric, resolution, metric > resolution)
            metric, coord_normals, faces_dict, n_new_edges = upsample_edges(coords_h5_fn, faces_h5_fn, faces_dict, new_edges_h5_fn, resolution, debug=debug,  n_new_edges=n_new_edges, coord_normals=coord_normals)
            counter+=1
            print_size(faces_dict)


        if coord_normals != [] :
            faces=fix_normals(faces,coords_h5['data'][:],coord_normals)

        del faces_dict 

def resample_gifti_to_h5(new_edges_h5_fn, reference_coords_h5_fn, input_list, output_list) :

    new_edges_h5 = h5py.File(new_edges_h5_fn, 'r')
    reference_coords_h5 = h5py.File(reference_coords_h5_fn, 'r')
    n = reference_coords_h5['data'].shape[0]
    n_edges = new_edges_h5['data'].shape[0]
    print('n=',n)
    for i, (in_fn, out_fn) in enumerate(zip(input_list,output_list)):

        if not os.path.exists(out_fn) :
            coords = nb.load(in_fn).agg_data('NIFTI_INTENT_POINTSET')
            n_coords = coords.shape[0]

            rsl_coords_h5 = h5py.File(out_fn, 'w')
            rsl_coords_h5.create_dataset('data', (n,3) )
            rsl_coords_h5['data'][ 0 : n_coords] = coords
            rsl_coords = rsl_coords_h5['data'][:]

            new_edges = new_edges_h5['data'][:].astype(int)
            for ii in range( n_edges) :
                #c0, c1 = new_edges_h5['data'][ii,:]
                #rsl_coords_h5['data'][ ii, : ] = (rsl_coords_h5['data'][int(c0),:] + rsl_coords_h5['data'][int(c1),:])/2.
                c0, c1 = new_edges[ii,:]
                rsl_coords[ n_coords + ii, : ] = (rsl_coords[int(c0),:] + rsl_coords[int(c1),:])/2.
                if ii % 100000 == 0 : 
                    print(100.*ii/n, n, c0, c1, rsl_coords[ii,:])

            rsl_coords_h5['data'][:] = rsl_coords
            rsl_coords_h5.close()


def upsample_gifti(input_fn,upsample_0_fn, upsample_1_fn, resolution, input_list=[], output_list=[], test=False, clobber=False, debug=False):

    faces_h5_fn = sub('.surf.gii','_new_faces.h5',upsample_0_fn)
    coords_h5_fn = sub('.surf.gii','_new_coords.h5',upsample_0_fn)
    new_edges_h5_fn = sub('.surf.gii','_new_edges.h5',upsample_0_fn)
    if not os.path.exists(coords_h5_fn) :
        print(os.path.exists(coords_h5_fn), coords_h5_fn)
        upsample_with_h5(input_fn, upsample_0_fn,  faces_h5_fn, coords_h5_fn, new_edges_h5_fn, resolution)

    if not os.path.exists(upsample_0_fn) :
        print(os.path.exists(upsample_0_fn), upsample_0_fn)
        write_gifti_from_h5(upsample_0_fn, coords_h5_fn, faces_h5_fn, input_fn ) 
    
    if input_list != []  and output_list != [] :
        print('new edges h5',new_edges_h5_fn)
        print('coords h5',coords_h5_fn)
        print('upsample 0', upsample_0_fn)
        resample_gifti_to_h5(new_edges_h5_fn, coords_h5_fn, input_list, output_list)

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
        print('Upsample fn', upsample_fn)

    if not os.path.exists(upsample_fn) or args.clobber or args.test  or True:
        upsample_gifti( args.input_fn, upsample_fn, float(args.resolution), input_list=args.input_list,  clobber=args.clobber, test=args.test, debug=args.debug)

