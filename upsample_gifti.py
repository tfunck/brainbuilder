import numpy as np
import os
import time
import sys
import nibabel as nib
import matplotlib.pyplot as plt
import h5py as h5
import tempfile
import argparse
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from vast.io_mesh import load_mesh_geometry, save_obj
from matplotlib_surface_plotting import plot_surf 
from c_upsample_mesh import upsample, resample
from re import sub
from utils.utils import shell,splitext
from glob import glob

def plot_faces(coords, faces, out_fn):
    patches=[]
    fig, ax = plt.subplots(1,1)
    for a,b,c in faces:
        print('-->',a,b,c)
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
    print('Plotting',out_fn)
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
    
    return d 

def save_gii(coords, triangles, reference_fn, out_fn):
    img = nib.load(reference_fn) 
    ar1 = nib.gifti.gifti.GiftiDataArray(data=coords.astype(np.float32), intent='NIFTI_INTENT_POINTSET') 
    ar2 = nib.gifti.gifti.GiftiDataArray(data=triangles.astype(np.int32), intent='NIFTI_INTENT_TRIANGLE') 
    out = nib.gifti.GiftiImage(darrays=[ar1,ar2], header=img.header, file_map=img.file_map, extra=img.extra, meta=img.meta, labeltable=img.labeltable) 
    out.to_filename(out_fn) 
    #print(out.print_summary())


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
    edge_face_idx = np.array( [ sorted_indices[ [i, i+1] ] for i in edges_idx ] )
    return edges, edge_face_idx

def upsample_edges(coords, faces, coord_normals,  faces_dict,  resolution, temp_alt_coords=None) :
    edges, edge_face_idx = get_edges_from_faces(faces) 
    e_0, e_1 = edges[:,0], edges[:,1]
    
    c_0 = coords[e_0] 
    c_1 = coords[e_1]

    ngh = get_ngh(faces)

    d_ij = calc_dist(c_0, c_1)

    long_edges = d_ij > resolution
    mm = np.argmax(long_edges)
    #long_edges =np.zeros_like(long_edges).astype(bool) 
    #long_edges[mm]=True
    new_coord_normals = calculate_new_coords(coord_normals, e_0, e_1, long_edges)
    new_coords = calculate_new_coords(coords, e_0, e_1, long_edges)

    new_alt_coords = None
    if temp_alt_coords != None:  
        for fn in temp_alt_coords.values() :
            alt_coords = np.load(fn+'.npy')
            new_alt_coords = calculate_new_coords(alt_coords, e_0, e_1, long_edges ) 
            np.save(fn, new_alt_coords)
    
    n_valid=np.sum(~long_edges)
    n_long=np.sum(long_edges)
    new_faces =np.zeros([ faces.shape[0]+n_long*2, 3 ]).astype(np.int32)
    new_faces[0:faces.shape[0]] = faces
    new_faces[faces.shape[0]:]=-1

    coord_offset = coords.shape[0]
    faces_offset = faces.shape[0] 
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
    edges_range = np.arange(edges.shape[0]).astype(int)
    new_faces_range = np.arange(new_faces.shape[0]).astype(int)
    #iterate over long edges (i.e., greater than desired resolution)
    list_intersect = lambda x, y : list(set([ii for ii in x+y if ii in x and ii in y]))

    for coord_idx, edge_counter in enumerate(edges_range[long_edges]):
        if coord_idx % 1000 ==0 : print('\t\t{}'.format(np.round(100*coord_idx/n_long,2)),end='\r')

        index = coord_offset + coord_idx

        
        a = edges[edge_counter][0]
        b = edges[edge_counter][1]
        ab_ngh = list_intersect(ngh[a],ngh[b]) 
        assert len(ab_ngh) == 2 , 'more than two neighbours for vertices {} {}'.format(a,b)

        c = ab_ngh[0]
        d = ab_ngh[1]
        face_idx=faces_dict[sorted_str([a,b,c])]
        #face_idx_0 = edge_face_idx[edge_counter][0]
        #face_idx_1 = edge_face_idx[edge_counter][1]
        #if a in new_faces[face_idx_0] and b in new_faces[face_idx_0] :
        #    face_idx = face_idx_0
        #elif a in new_faces[face_idx_1] and b in new_faces[face_idx_1] :
        #    face_idx = face_idx_1
        #    print('using second face')
        #else :
        #assert a in new_faces[face_idx] and b in new_faces[face_idx], 'a {} and b {} are not in new face {}'.format(a,b,new_faces[face_idx_0], new_faces[face_idx_1])

        #temp_ar = [ i for i in new_faces[face_idx] if i != a and i != b ]

        #assert len(temp_ar) == 1, 'Error, should only have 1 {} a {} b {} -- {}'.format(temp_ar,a,b,new_faces[face_idx])
        #c=temp_ar[0]

        #ab_ngh = [ i for i in list_intersect(ngh[a],ngh[b]) if i not in new_faces[face_idx] ]
        #print('coords a', a, new_coords[a])
        #print('coords b', b, new_coords[b])
        #print('coords c', c, new_coords[c])

        #assert len(ab_ngh) == 1 , 'more than two neighbours for vertices {} {}'.format(a,b)
        #d = ab_ngh[0]
        #print('coords d', new_coords[d])
        # print('coords index', index, new_coords[index])
        
        #print('a ',a,'\tb ', b,'\tc ', c, '\td', d, '\ti ', index)
        #assert a in new_faces[face_idx] and b in new_faces[face_idx] , 'Error, edges not in face {} {} {} {} {}'.format(a,b,faces[face_idx], a in new_faces[face_idx], b in new_faces[face_idx])

        #a,b,c = new_faces[face_idx]
        
        ar = list_intersect(ngh[a], ngh[b])
        #print('opposing face', a,b,d)
        opposite_poly_index=faces_dict[sorted_str([a,b,d])]
        #print(face_idx, opposite_poly_index)
        #print('\topposing face-->',a,b,d, new_faces[opposite_poly_index], new_faces[edge_face_idx[edge_counter][1] ] )

        #the idea is that the new coordinate that was interpolated between vertex a and b is stored
        #in new_coords[index]
        #print('ngh c',c,':', ngh[c])
        #print('ngh d',d,':', ngh[d])
        #print(a,b,c,d,face_idx)
        assert len(list_intersect(ngh[a],ngh[b])) == 2, 'a b index'
        assert len(list_intersect(ngh[a],ngh[c])) == 2, 'a c index'
        assert len(list_intersect(ngh[a],ngh[d])) == 2, 'a d index'
        assert len(list_intersect(ngh[c],ngh[d])) == 2, 'c {} d {} index {}, {}, {}'.format(c,d,ngh[c], ngh[d], list_intersect(ngh[c],ngh[d]) )
        #insert new vertex in mesh --> 4 new polygons
        #print('current face',new_faces[face_idx] )

        new_faces[face_idx] = sorted([a,index,c]) 
        new_faces[opposite_poly_index] = sorted([index,c,b])

        edge_face_idx[ edge_face_idx == face_idx ] = face_idx

        new_faces[faces_offset] = sorted([a,index,d]) 
        new_faces[faces_offset+1] = sorted([index,b,d])
        print()
        faces_dict[sorted_str([a,index,c])] = face_idx
        faces_dict[sorted_str([index,c,b])] = opposite_poly_index
        faces_dict[sorted_str([a,index,d])] = faces_offset
        faces_dict[sorted_str([index,b,d])] = faces_offset+1
        
        #for var in [face_idx, opposite_poly_index, faces_offset, faces_offset+1] :
        #    print(new_faces[var], '-->',var )
        
        #print('ngh[c]', ngh[c])
        #update the neighbours of each vertex to reflect changes to mesh
        ngh[a] = [ ii if ii != b else index for ii in ngh[a] ] 
        ngh[b] = [ ii if ii != a else index for ii in ngh[b] ]
        ngh[c].append(index) 
        ngh[d].append(index) 

        ngh[index] = [a,b,c,d]
        #print('\t\tHEYO', new_faces[31744])
        #print('face idx and opp',face_idx, opposite_poly_index)
        #for nn, f in enumerate(new_faces):
        #    if a in f and b in f :
        #        print('Error should not longer have polygon with',a,'and', b,' in it',f,'for index',nn)
        #        exit(0)
        #print('ngh[c]', ngh[c])
        assert len(list_intersect(ngh[a],ngh[c])) == 2, 'a c'
        assert len(list_intersect(ngh[c],ngh[b])) == 2, 'c b {} {} {}'.format(ngh[c],ngh[b], list_intersect(ngh[c],ngh[b]))
        assert len(list_intersect(ngh[a],ngh[d])) == 2, 'a d'
        assert len(list_intersect(ngh[b],ngh[d])) == 2, 'b d'
        assert len(list_intersect(ngh[a],ngh[index])) == 2, 'a index'
        assert len(list_intersect(ngh[b],ngh[index])) == 2, 'b index'
        assert len(list_intersect(ngh[c],ngh[index])) == 2, 'c index'
        assert len(list_intersect(ngh[d],ngh[index])) == 2, 'd index'

        faces_offset += 2

        #get_edges_from_faces(new_faces[new_faces[:,0] != -1]) 
    return new_coords, new_faces, new_coord_normals, faces_dict

def load_alt_coordinates(alt_input_list):
    n_alt = len(alt_input_list)
    temp_alt_fn={}
    #f = h5.File(f'{alt_h5_fn}','w')
    for i, fn in enumerate(alt_input_list) :
        alt_coords = nib.load(fn).agg_data('NIFTI_INTENT_POINTSET')
        temp_fn='{}{}'.format(tempfile.NamedTemporaryFile().name,os.path.basename(sub('.surf.gii','',fn)))
        np.save(temp_fn, alt_coords)
        temp_alt_fn[fn] = temp_fn
    print(temp_alt_fn)
    return temp_alt_fn

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
    ext = splitext(upsample_fn)[1]
    if ext == '.surf.gii' :
        save_gii( coords, faces, input_fn, upsample_fn )
    elif ext == '.obj' :
        save_obj(upsample_fn,coords,faces)
    else :
        print('not implemented for ext', ext)
        exit(1)

def write_outputs(upsample_fn, coords, faces, input_fn, temp_alt_coords ) :
    print('\tWriting',upsample_fn)

    write_mesh( coords, faces, input_fn, upsample_fn )
    if temp_alt_coords != None :
        for orig_fn, fn in temp_alt_coords.items() :
            alt_upsample_fn=sub('.surf.gii','_rsl.surf.gii',orig_fn)
            print('\tWriting',alt_upsample_fn)
            write_mesh(np.load(fn+'.npy'), faces, input_fn, alt_upsample_fn )

def upsample_gifti(input_fn,upsample_fn,  resolution, alt_input_list=[],test=False, clobber=False):

    if not os.path.exists(upsample_fn) or clobber :
        
        if test :
            faces=np.array([[0,1,5], 
                            [0,3,5], 
                            [2,3,5], 
                            [1,2,5],
                            [0,1,4],
                            [0,3,4],
                            [2,3,4],
                            [1,2,4]] 
                            ).astype(int) 
            coords=np.array([[0,0,0],
                            [0,1,0],
                            [1,1,0],
                            [1,0,0],
                            [.5,.5,1],
                            [.5,.5,-1]]).astype(float)

            write_outputs('test.obj', coords, faces, input_fn, None ) 
            upsample_fn='test_rsl.obj'
        else :
            ext = os.path.splitext(input_fn)[1]
            if ext == '.gii' :
                mesh = nib.load(input_fn)
                faces = mesh.agg_data('NIFTI_INTENT_TRIANGLE')
                coords= mesh.agg_data('NIFTI_INTENT_POINTSET')
            elif ext == '.obj' :
                mesh_dict = load_mesh_geometry(input_fn)
                coords = mesh_dict['coords']
                faces = mesh_dict['faces']
            else :
                print('Error: could not recognize filetype from extension,',ext)
                exit(1)
        #calculate surface normals
        normals = np.array([ np.cross(coords[b]-coords[a],coords[c]-coords[a]) for a,b,c in faces ])

        faces = np.sort(faces, axis=1).astype(np.int32)
        faces_dict={}
        for i, (x,y,z) in enumerate(faces):
            key=sorted_str([x,y,z])
            faces_dict[key] = i
      
        alt_h5_f = None
        temp_alt_coords=None
        if alt_input_list != [] :
            temp_alt_coords = load_alt_coordinates(alt_input_list)
        
        coord_normals=setup_coordinate_normals(faces,normals,coords)
        
        #calculate edge lengths
        d = [   calc_dist(coords[faces[:,0]], coords[faces[:,1]]),
                calc_dist(coords[faces[:,1]], coords[faces[:,2]]),
                calc_dist(coords[faces[:,2]], coords[faces[:,1]])]

        max_len = np.round(np.max(d),2)
        avg_len = np.round(np.mean(d),2)
        perc95 = np.round(np.percentile(d, [95])[0],2)
        print('resolution', resolution)
        print('\tmax edge',max_len,'95th percentile',perc95,'\tavg', avg_len,'\tcoords=', coords.shape[0], '\tfaces=', faces.shape[0] )
        counter=0
        start_resolution = max(int(max_len), resolution) 
        for target_resolution in np.arange(start_resolution, resolution-1,-1) :
            print('target reslution:', target_resolution)
            while max_len > target_resolution and max_len > resolution :
                
                coords, faces, coord_normals, faces_dict = upsample_edges(coords, faces, coord_normals, faces_dict, target_resolution,temp_alt_coords=temp_alt_coords)
                counter+=1
                d = [   calc_dist(coords[faces[:,0]], coords[faces[:,1]]),
                        calc_dist(coords[faces[:,1]], coords[faces[:,2]]),
                        calc_dist(coords[faces[:,2]], coords[faces[:,0]])]
                max_len = np.round(np.max(d),2)
                avg_len = np.round(np.mean(d),2)
                perc95 = np.round(np.percentile(d, [95])[0],2)
                print('\tmax edge',max_len,'95th percentile',perc95,'\tavg', avg_len,'\tcoords=', coords.shape[0], '\tfaces=', faces.shape[0] )
        print('\tFinal coords=', coords.shape, 'starting faces', faces.shape)

        faces=fix_normals(faces,coords,coord_normals)

        write_outputs(upsample_fn, coords, faces, input_fn, temp_alt_coords ) 
    

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', dest='input_fn',  default=None, help='input fn')
    parser.add_argument('-r', dest='resolution',  default=None, help='resolution')
    parser.add_argument('-a', dest='alt_input_list', nargs='+', default=[], help='optional fn')
    parser.add_argument('-c', action='store_true', default=False, dest='clobber', help='optional fn')
    parser.add_argument('-t', action='store_true', default=False, dest='test', help='test upsampling')
    args = parser.parse_args()
    
    upsample_fn=''
    if not args.test:
        basename, ext = splitext(args.input_fn)
        upsample_fn = f'{basename}_rsl{ext}' 
        print('Upsample fn', upsample_fn)

    if not os.path.exists(upsample_fn) or args.clobber or args.test :
        upsample_gifti( args.input_fn, upsample_fn, float(args.resolution), alt_input_list=args.alt_input_list,  clobber=args.clobber, test=args.test)
