import numpy as np
import os
import sys
import nibabel as nib
from c_upsample_mesh import upsample, resample
from re import sub
from utils.utils import shell
from glob import glob

def read_coords(fn):
    lines_list=[]
    print("reading",fn)
    with open(fn,'r') as F :
        for line_i , l in enumerate(F.readlines()) :
            lines_list.append(l.rstrip().split(","))
    ar = np.array(lines_list)
    coords = ar[ ar[:,0]=='v'][:,1:4].astype(float)
    root_index = ar[ ar[:,0]=='r'][:,1:4].astype(int)
    faces = ar[ ar[:,0]=='p' ][:,1:4].astype(int)

    return np.array(coords), np.array(root_index), faces.astype(np.double)
def add_entry(d,a,b,c):
    try :
        d[a] += [b,c] 
    except KeyError :
        d[a] = [b,c]
    return d
def get_ngh(triangles):
    d={}
    for i,j,k in triangles:
        d = add_entry(d,i,j,k)
        d = add_entry(d,j,i,k)
        d = add_entry(d,k,j,i)

    for key in d.keys():
        d[key] =np.unique(d[key])
    
    return d 

def save_gii(coords, triangles, reference_fn, out_fn):
    img = nib.load(reference_fn) 
    ar1 = nib.gifti.gifti.GiftiDataArray(data=coords, intent='NIFTI_INTENT_POINTSET') 
    ar2 = nib.gifti.gifti.GiftiDataArray(data=triangles, intent='NIFTI_INTENT_TRIANGLE') 
    out = nib.gifti.GiftiImage(darrays=[ar1,ar2], header=img.header, file_map=img.file_map, extra=img.extra, meta=img.meta, labeltable=img.labeltable) 
    out.to_filename(out_fn) 
    #print(out.print_summary())

def upsample_gifti(upsample_fn, input_fn, resolution, clobber=False, upsample_csv_fn=None):
    
    if not os.path.exists(upsample_fn) or clobber :
        #flatten ngh list to pass to c cod 
        mesh = nib.load(input_fn)
        coords= mesh.agg_data('NIFTI_INTENT_POINTSET')
        faces = mesh.agg_data('NIFTI_INTENT_TRIANGLE')
        
        nvtx = int(coords.shape[0])
        npoly = int(faces.shape[0])

        upsample(
                np.array(coords).flatten().astype(np.float32), 
                faces.flatten().astype(np.int32), 
                upsample_csv_fn, float(resolution), 
                nvtx,
                npoly)

        #convert the upsampled .csv points into a proper gifti surfer
        coords_rsl,root_index, faces_rsl = read_coords(upsample_csv_fn)
        print(faces_rsl.shape, coords_rsl.shape, np.max(faces_rsl))
        save_gii( coords_rsl, faces_rsl, input_fn, upsample_fn )

    return upsample_csv_fn

def resample_to_reference(low_res_fn, high_res_fn, out_fn):
    #flatten ngh list to pass to c cod 
    print("Resampling to", high_res_fn)
    if os.path.splitext(high_res_fn)[0] =='csv' : 
        print("Error: should be csv but got", high_res_fn)
        exit(1)
    
    tgt_faces = nib.load(high_res_fn).agg_data('NIFTI_INTENT_TRIANGLE')
    tgt_coords = nib.load(high_res_fn).agg_data('NIFTI_INTENT_POINTSET')
    
    img = nib.load(low_res_fn) 
    coords = img.agg_data('NIFTI_INTENT_POINTSET') 
    print('\trsl n vertices = ', tgt_coords.shape[0])

    #upsample surface and save in an intermediate .csv file
    sphere_rsl_csv_fn = os.path.splitext(low_res_fn)[0] + '.csv'
    resampled_coords = np.zeros_like(tgt_coords)
    resampled_coords[0:coords.shape[0],:] = tgt_coords[0:coords.shape[0]]
    print(np.mean(np.abs(tgt_coords[0:coords.shape[0]]-coords),axis=0))
    exit(0)

    #resampled_coords[0:coords.shape[0],:] = coords
    index_list =range(coords.shape[0], tgt_coords.shape[0]) 
    
    ngh = get_ngh(tgt_faces)
    while len(index_list) > 0  :
        new_index_list=[]
        for i in index_list :
            cur_ngh_idx = ngh[i]
            ngh_coords = resampled_coords[cur_ngh_idx]

            idx = (ngh_coords != np.array([0,0,0])).all(axis=1)
            
            ngh_tgt_coords = tgt_coords[cur_ngh_idx][idx]
            central_tgt_coord = tgt_coords[i]
            distances = [ np.sqrt(np.sum(np.power(x-central_tgt_coord,2))) for x in ngh_tgt_coords ]
            weights = distances/ np.sum(distances)

            new_coords = tgt_coords[i] # np.sum( np.multiply( ngh_coords[idx] , weights.reshape(-1,1)),axis=0)
            #print('ngh')
            #print(ngh[i])
            #print('weight')
            #print(weights)
            #print('ngh')
            print(ngh_coords)
            #print('new')
            print('a',new_coords)
            print('b',tgt_coords[i])
            print()
            #if i > index_list[100] :
            #    exit(1)
            #if np.sum(idx) < 2 or True in (new_coords ==  ngh_coords).all(axis=0) : 
            #    new_index_list.append(i)
            #else :
                
            resampled_coords[i,:] =  new_coords

        index_list=new_index_list
        print('\t\tn skipped ', len(index_list))


    ar_unq, idx_unq, count = np.unique( resampled_coords[np.sum(resampled_coords,axis=1) > 0], axis=0, return_index=True, return_counts=True)
    print(np.column_stack([ count[count>1],ar_unq[count>1]]))
    #convert the upsampled .csv points into a proper gifti surfer

    #sphere_coords_rsl = nib.load(high_res_fn).agg_data('NIFTI_INTENT_POINTSET') 
    check_unique( resampled_coords  )
    save_gii( resampled_coords, tgt_faces, high_res_fn, out_fn )

    return 0

def check_unique(coords) :
    ar_unq, count = np.unique( coords , axis=0, return_counts=True)
    print(ar_unq.shape, coords.shape)
    assert ar_unq.shape == coords.shape, 'Error: mesh inflation produces mesh with duplicates'

def create_high_res_sphere(input_fn, upsample_fn, sphere_fn, sphere_rsl_fn, resolution, clobber=False, optional_reference=None, upsample_csv_fn=None):
    nvtx = nib.load(input_fn).agg_data('NIFTI_INTENT_POINTSET').shape[0]
    gifti_example_fn=input_fn
    avg_edge_len = 0.75 # 0.2 if resolution/2. > 0.2 else resolution/2.

    if not os.path.exists(upsample_fn) or clobber :
        if optional_reference == None :
            shell(f'~/freesurfer/bin/mris_remesh -i {input_fn} -o {upsample_fn} --edge-len {avg_edge_len} --iters 1' )
        else :
            resample_to_reference(input_fn, optional_reference , upsample_fn)

    if not os.path.exists(sphere_fn) or  clobber :
        print('\tInflate to sphere')
        shell('~/freesurfer/bin/mris_inflate -n 30  {} {}'.format(input_fn, sphere_fn))
    
    if not os.path.exists(sphere_rsl_fn) or clobber :
        print("Resample to reference")
        resample_to_reference( sphere_fn, upsample_fn, sphere_rsl_fn)

if __name__ == "__main__" :
    if len(sys.argv) != 3 or sys.argv[1] == '-h'  :
        print("useage: upsample_gifti input.surf.gii resolution output.surf.gii")
        exit(1)

    input_fn = sys.argv[1]
    resolution = sys.argv[2]
    upsample_fn = sub('.surf.gii','_upsample.surf.gii',input_fn)
    sphere_fn = sub('.surf.gii','_sphere.surf.gii',input_fn)
    sphere_rsl_fn = sub('.surf.gii','_sphere_rsl.surf.gii',input_fn)
    print('Upsample fn', upsample_fn)
    create_high_res_sphere(input_fn, upsample_fn, sphere_fn, sphere_rsl_fn, resolution)
