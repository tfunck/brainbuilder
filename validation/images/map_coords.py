import h5py 
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label


def get_section_intervals(vol):

    valid_sections = np.sum(vol, axis=(0,2)) > 0
    
    labeled_sections, nlabels = label(valid_sections)

    intervals = [ (np.where(labeled_sections==i)[0][0], np.where(labeled_sections==i)[0][-1]) for i in range(1, nlabels) ]
    
    return intervals
    

def get_valid_coords(vol, coords, iv, iw):
    valid_coords_idx = (coords[:,1] >= iw[0]) & (coords[:,1]<=iw[1])
    valid_coords_idx = valid_coords_idx.reshape(valid_coords_idx.shape[0])
    valid_coords = coords[valid_coords_idx, :]
    return valid_coords, valid_coords_idx


def project_volumes_to_surfaces(surf_fn_list, vol_fn_list, interp_csv):
    nvertices = h5py.File(surf_fn_list[0],'r')['data'].shape[0]
    all_values=np.zeros(nvertices)

    for i, (surf_fn, vol_fn) in enumerate(zip(surf_fn_list, vol_fn_list)) :
        print('\tSlab =',i+1)
        coords_h5 = h5py.File(surf_fn,'r')
        coords = coords_h5['data']

        img = nib.load(vol_fn)
        vol = img.get_fdata()

        xstart = img.affine[0,3]
        ystart = img.affine[1,3]
        zstart = img.affine[2,3]
        xstep = img.affine[0,0]
        ystep = img.affine[1,1]
        zstep = img.affine[2,2]

        intervals_voxel = get_section_intervals(vol)
        intervals_world = np.array(intervals_voxel) * ystep + ystart

        for iv, iw in zip(intervals_voxel, intervals_world):
            section = vol[:,iv[0],:]

            valid_coords_world, valid_coords_idx = get_valid_coords(vol, coords, iv, iw)

            x = (valid_coords_world[:,0]-xstart)/xstep
            z = (valid_coords_world[:,2]-zstart)/zstep 

            values = section[np.rint(x).astype(int),np.rint(z).astype(int)]

            assert np.mean(values) > 0, 'Error: empty section'

            all_values[valid_coords_idx] = values

            print('\t\tvoxel:', iv, '\tworld:', iw, '\twidth', np.round(iw[1]-iw[0],3)  )
            #plt.figure(figsize=(24,24))
            #plt.imshow(section)
            #plt.scatter(z,x,s=1,c='r',marker='.')
            #plt.savefig(f'{iv[0]}_{iv[1]}.png', dpi=400)
            #print('\tSaved', f'{iv[0]}_{iv[1]}.png')
            #plt.clf()
            #plt.cla()
    np.savetxt(interp_csv, all_values)
        

vol_fn_list = ['thickened_1_oxot_0.6.nii.gz','thickened_2_oxot_0.6.nii.gz', 'thickened_3_oxot_0.6.nii.gz','thickened_4_oxot_0.6.nii.gz','thickened_4_oxot_0.6.nii.gz', 'thickened_6_oxot_0.6.nii.gz' ]
surf_fn_list=['slab-1_surf_0.6mm_0.4_rsl.h5', 'slab-2_surf_0.6mm_0.4_rsl.h5', 'slab-3_surf_0.6mm_0.4_rsl.h5', 'slab-4_surf_0.6mm_0.4_rsl.h5', 'slab-5_surf_0.6mm_0.4_rsl.h5', 'slab-6_surf_0.6mm_0.4_rsl.h5']

project_volumes_to_surfaces(surf_fn_list, vol_fn_list, 'test_all_values.csv')
