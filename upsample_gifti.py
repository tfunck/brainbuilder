import numpy as np
import os
import time
import sys
import nibabel as nib
import matplotlib.pyplot as plt
from c_upsample_mesh import upsample, resample
from re import sub
from utils.utils import shell
from glob import glob
import argparse

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

def upsample_edges(coords, faces, faces_dict, i,j,k , resolution, alt_coords) :
    f_i = faces[:,i]
    f_j = faces[:,j]
    #f_k = faces[:,k]

    c_i = coords[f_i] 
    c_j = coords[f_j] 
    #c_k = coords[f_k] 

    ngh = get_ngh(faces)

    d_ij = calc_dist(c_i, c_j)

    long_edges = d_ij > resolution
    f_i = faces[:,i]

    new_coords = (c_i[long_edges] + c_j[long_edges])/2.
    new_coords = np.concatenate([coords, new_coords])

    new_alt_coords = None
    if type(alt_coords) == type(np.array([])) :
        ac_i = alt_coords[f_i] 
        ac_j = alt_coords[f_j] 
        new_alt_coords = (ac_i[long_edges] + ac_j[long_edges])/2.
        new_alt_coords = np.concatenate([alt_coords, new_alt_coords])
    
    f_i = faces[:,i]
    n_valid=np.sum(~long_edges)
    n_long=np.sum(long_edges)
    new_faces =np.zeros([ faces.shape[0]+n_long*2, 3 ]).astype(np.int32)
    new_faces[0:faces.shape[0]] = faces
    new_faces[faces.shape[0]:]=-1

    f_i = faces[:,i]
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
    faces_range = np.arange(faces.shape[0]).astype(int)
    new_faces_range = np.arange(new_faces.shape[0]).astype(int)
    #iterate over polygons/faces/triangles that have a long edge (i.e., greater than desired resolution)
    #between the ith and jth element of the triangle.
    for coord_idx, counter in enumerate(faces_range[long_edges]):
        if coord_idx % 1000 ==0 : 
            print('\t\t{}'.format(np.round(100*coord_idx/n_long,2)),end='\r')
            #print(np.round(time_0,3),np.round(time_1,3),np.round(time_2,3))
            #print('\t', np.round(time_2a,4), np.round(time_2b,4),np.round(time_2c,4))
        a,b,c = new_faces[counter]
        
        index = coord_offset + coord_idx
        
        list_intersect = lambda x, y : list(set([ii for ii in x+y if ii in x and ii in y]))
        ar = list_intersect(ngh[a],ngh[b])
        ar.remove(c)
        ar = [ii for ii in ar if ii != c] # ar[ ar != c ]
        len_ar = len(ar)
        if len_ar == 0 : continue
        assert len_ar == 1, f'Error: {a} and {b} should only have one value but has {ar}\n{ngh[a]}\n{ngh[b]}'
        d = ar[0] 

        ar = list_intersect(ngh[a], ngh[b])

        opposite_poly_index=faces_dict[sorted_str([a,b,d])]

        #the idea is that the new coordinate that was interpolated between vertex a and b is stored
        #in new_coords[index]

        #insert new vertex in mesh --> 4 new polygons
        new_faces[counter] = sorted([a,index,c]) 
        new_faces[opposite_poly_index] = sorted([index,c,b]) 
        new_faces[faces_offset] = sorted([a,index,d]) 
        new_faces[faces_offset+1] = sorted([index,b,d])
        faces_dict[sorted_str([a,index,c])] = counter 
        faces_dict[sorted_str([index,c,b])] = opposite_poly_index
        faces_dict[sorted_str([a,index,d])] = faces_offset
        faces_dict[sorted_str([index,b,d])] = faces_offset+1
        
        #update the neighbours of each vertex to reflect changes to mesh
        ngh[a] = [ ii if ii != b else index for ii in ngh[a] ] 
        ngh[b] = [ ii if ii != a else index for ii in ngh[b] ]
        ngh[c].append(index) 
        ngh[d].append(index) 
        ngh[index]=[a,b,c,d]


        faces_offset += 2

    new_faces = np.unique(new_faces, axis=1)
    return new_coords, new_faces, faces_dict, new_alt_coords


def upsample_gifti(input_fn,upsample_fn,  resolution, alt_input_fn=False,alt_upsample_fn=None, clobber=False):

    if not os.path.exists(upsample_fn) or clobber :
        #flatten ngh list to pass to c cod 
        mesh = nib.load(input_fn)
        faces = mesh.agg_data('NIFTI_INTENT_TRIANGLE')
        coords= mesh.agg_data('NIFTI_INTENT_POINTSET')

        faces = np.sort(faces, axis=1).astype(np.int32)
        faces_dict={}
        for i, (x,y,z) in enumerate(faces):
            faces_dict[sorted_str([x,y,z])] = i
       
        alt_coords=None
        if alt_input_fn != None :
            alt_coords = nib.load(alt_input_fn).agg_data('NIFTI_INTENT_POINTSET')
        
        average_normal = np.sum([ np.cross(coords[b]-coords[a],coords[c]-coords[a]) for a,b,c in faces ],axis=0)
        
        #calculate edge lengths
        d = [   calc_dist(coords[faces[:,0]], coords[faces[:,1]]),
                calc_dist(coords[faces[:,1]], coords[faces[:,2]]),
                calc_dist(coords[faces[:,2]], coords[faces[:,1]])]
        max_len = np.round(np.max(d),2)
        avg_len = np.round(np.mean(d),2)
        perc95 = np.round(np.percentile(d, [95])[0],2)
        print('resolution', resolution)
        print('\tmax edge',max_len,'95th percentile',perc95,'\tavg', avg_len,'\tcoords=', coords.shape[0], '\tfaces=', faces.shape[0] )
        start_resolution = max(int(max_len), resolution) 
        for target_resolution in np.arange(start_resolution, resolution-1,-1) :
            while perc95 > target_resolution and max_len > resolution :
                coords, faces, faces_dict, alt_coords = upsample_edges(coords, faces, faces_dict, 0, 1, 2, target_resolution,alt_coords=alt_coords)
                coords, faces, faces_dict, alt_coords = upsample_edges(coords, faces, faces_dict, 1, 2, 0, target_resolution,alt_coords=alt_coords)
                coords, faces, faces_dict, alt_coords = upsample_edges(coords, faces, faces_dict, 2, 1, 0, target_resolution,alt_coords=alt_coords)
                d = [   calc_dist(coords[faces[:,0]], coords[faces[:,1]]),
                        calc_dist(coords[faces[:,1]], coords[faces[:,2]]),
                        calc_dist(coords[faces[:,2]], coords[faces[:,1]])]
                max_len = np.round(np.max(d),2)
                avg_len = np.round(np.mean(d),2)
                perc95 = np.round(np.percentile(d, [95])[0],2)
                print('\tmax edge',max_len,'95th percentile',perc95,'\tavg', avg_len,'\tcoords=', coords.shape[0], '\tfaces=', faces.shape[0] )

        print('\tFinal coords=', coords.shape, 'starting faces', faces.shape)
        for i in range(faces.shape[0]) :
            a,b,c = faces[i]
            test_normal = np.cross(coords[b]-coords[a], coords[c]-coords[a])
            x=np.dot(average_normal, test_normal)
            if x < 0 : faces[i]=[c,b,a]

        print('\tWriting',upsample_fn)
        save_gii( coords, faces, input_fn, upsample_fn )
        if type(alt_coords) == type(np.array([])) :
            print('\tWriting',alt_upsample_fn)
            save_gii( alt_coords, faces, input_fn, alt_upsample_fn )




def create_high_res_sphere(input_fn, upsample_fn, sphere_fn, sphere_rsl_fn, resolution, clobber=False ):
    if not os.path.exists(sphere_fn) or  clobber :
        print('\tInflate to sphere')
        shell('~/freesurfer/bin/mris_inflate -n 200  {} {}'.format(input_fn, sphere_fn))
        #shell('mris_inflate -n 200  {} {}'.format(input_fn, sphere_fn))
    resolution=1
    if not os.path.exists(upsample_fn) or clobber :
        upsample_gifti( input_fn, upsample_fn, float(resolution), alt_input_fn=sphere_fn,  alt_upsample_fn=sphere_rsl_fn, clobber=clobber)


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', dest='input_fn',  default=None, help='input fn')
    parser.add_argument('-r', dest='resolution',  default=None, help='resolution')
    parser.add_argument('-a', dest='alt_input_fn',  default=None, help='optional fn')
    parser.add_argument('-c', action='store_true', default=False, dest='clobber', help='optional fn')
    args = parser.parse_args()


    upsample_fn = sub('.surf.gii','_rsl.surf.gii',args.input_fn)
    sphere_fn = sub('.surf.gii','_sphere.surf.gii',args.input_fn)
    sphere_rsl_fn = sub('.surf.gii','_sphere_rsl.surf.gii',args.input_fn)
    print('Upsample fn', upsample_fn)
    create_high_res_sphere(args.input_fn, upsample_fn, sphere_fn, sphere_rsl_fn, args.resolution,clobber=args.clobber)
