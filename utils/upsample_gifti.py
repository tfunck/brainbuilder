import numpy as np
import os
import sys
import nibabel as nib
from c_upsample_mesh import upsample, resample
from re import sub
from utils import shell

def read_coords(fn):
    lines_list=[]
    with open(fn,'r') as F :
        for line_i , l in enumerate(F.readlines()) :
            lines_list.append(l.rstrip().split(","))
    ar = np.array(lines_list)
    coords = ar[ ar[:,0]=='v'][:,1:4].astype(float)
    root_index = ar[ ar[:,0]=='r'][:,1:4].astype(int)
    faces = ar[ ar[:,0]=='p' ][:,1:4].astype(int)

    return np.array(coords), np.array(root_index), faces.astype(np.double)

def save_gii(coords, triangles, reference_fn, out_fn):
    img = nib.load(reference_fn) 
    ar1 = nib.gifti.gifti.GiftiDataArray(data=coords, intent='NIFTI_INTENT_POINTSET') 
    ar2 = nib.gifti.gifti.GiftiDataArray(data=triangles, intent='NIFTI_INTENT_TRIANGLE') 
    out = nib.gifti.GiftiImage(darrays=[ar1,ar2], header=img.header, file_map=img.file_map, extra=img.extra, meta=img.meta, labeltable=img.labeltable) 
    out.to_filename(out_fn) 
    #print(out.print_summary())

def upsample_gifti(upsample_fn, input_fn, resolution, clobber=False):
    
    #upsample surface and save in an intermediate .csv file
    upsample_csv_fn = os.path.splitext(upsample_fn)[0] + '.csv'
    if not os.path.exists(upsample_fn) or not os.path.exists(upsample_csv_fn) or clobber :
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

def resample_to_reference(sphere_rsl_fn, resample_fn, new_coords, reference_csv_fn, lores_nvtx):
    #flatten ngh list to pass to c cod 
    coords, root_index, faces = read_coords(reference_csv_fn);

    #upsample surface and save in an intermediate .csv file
    sphere_rsl_csv_fn = os.path.splitext(sphere_rsl_fn)[0] + '.csv'
    
    resample(
            np.array(new_coords).flatten().astype(np.float32), 
            root_index.flatten().astype(np.int32), 
            sphere_rsl_csv_fn,
            coords.shape[0],
            lores_nvtx
            )

    #convert the upsampled .csv points into a proper gifti surfer
    sphere_coords_rsl, [], [] = read_coords(sphere_rsl_csv_fn)
    print("-->",sphere_coords_rsl.shape, coords.shape)
    save_gii( sphere_coords_rsl, faces, resample_fn, sphere_rsl_fn )

    return 0

def create_high_res_sphere(input_fn, upsample_fn, sphere_fn, sphere_rsl_fn, resolution, clobber=False):
    nvtx = nib.load(input_fn).agg_data('NIFTI_INTENT_POINTSET').shape[0]

    print("\tUpsample")
    upsample_csv_fn = upsample_gifti(upsample_fn, input_fn, resolution, clobber=clobber)
    
    print('\tInflate to sphere')
    if not os.path.exists(sphere_fn) or clobber :
        #inflate surface to sphere using freesurfer software
        shell('mris_inflate -n 100  {} {}'.format(input_fn, sphere_fn))
    
    print("Resample to reference")
    if not os.path.exists(sphere_rsl_fn) or clobber :
        sphere_coords = nib.load(sphere_fn).agg_data('NIFTI_INTENT_POINTSET')
        resample_to_reference( sphere_rsl_fn,  input_fn, sphere_coords, upsample_csv_fn, nvtx )

if __name__ == "__main__" :
    if len(sys.argv) != 3 or sys.argv[1] == '-h'  :
        print("useage: upsample_gifti input.surf.gii resolution output.surf.gii")
        exit(1)

    input_fn = sys.argv[1]
    resolution = sys.argv[2]
    upsample_fn=sub('.surf.gii','_upsample.surf.gii',input_fn)
    sphere_fn = sub('.surf.gii','_sphere.surf.gii',input_fn)
    sphere_rsl_fn = sub('.surf.gii','_sphere_rsl.surf.gii',input_fn)
    create_high_res_sphere(input_fn, upsample_fn, sphere_fn, sphere_rsl_fn, resolution)
