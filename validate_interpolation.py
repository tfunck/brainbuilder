import nibabel as nib
import numpy as np
import stripy
import pandas as pd
import vast.surface_tools as surface_tools
import argparse
from glob import glob

def read_gifti(mesh_fn):
    mesh = nib.load(mesh_fn)

    coords = mesh.agg_data('NIFTI_INTENT_POINTSET')
    faces =  mesh.agg_data('NIFTI_INTENT_TRIANGLE')
    return coords, faces.astype(int)

def get_ngh(coords, faces):
    ngh={}
    for i0, i1, i2 in faces :
        def f(ngh, i,j,k):
            try :
                ngh[i] += [j,k]
            except KeyError :
                ngh[i] = [j,k]
            return ngh
        ngh = f(ngh, i0, i1, i2)
        ngh = f(ngh, i1, i0, i2)
        ngh = f(ngh, i2, i0, i1)
    
    for key, value in ngh.items() :
        ngh[key] = np.unique(value)
    
    return ngh

def calculate_neighbours(i, ngh, coords, densities, max_depth) :
    ngh_core = [i]
    ngh_list = ngh[i]
    dist_list = []
    for depth in range(max_depth):
        print('\tdepth', max_depth, depth)
        new_ngh=[]
        print(i,ngh_list)
        border=[]
        dist_list=[]
        for ii in ngh_list :
            if densities[ii] > 0 :
                new_ngh += list(ngh[ii])
                border.append(ii)
                if depth == max_depth:
                    dist_list.append(np.mean(np.abs(coords[i] - coords[ii])))

        if depth < max_depth-1:
            ngh_core += border 

        ngh_list = new_ngh
    
    dist = np.mean(dist_list)
    return ngh_core, border, dist

def interpolate_over_sphere(densities, coords_all, faces, idx, border, i):
    spherical_coords = surface_tools.spherical_np(coords_all)
    coords_src = spherical_coords[border]
    coords = spherical_coords[idx+border]
    
    lats, lons = coords[:,1]-np.pi/2, coords[:,2]
    lats_src, lons_src = coords_src[:,1]-np.pi/2, coords_src[:,2]

    mesh = stripy.sTriangulation(lons_src, lats_src)

    # interpolate over the sphere
    interp_val, interp_type = mesh.interpolate(lons,lats, zdata=densities[border], order=1)
    #print(interp_val) 
    #print(densities[idx])
    
    return interp_val[0] / densities[idx] 

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--out-dir', dest='out_dir', default='/scratch/tfunck/output2/5_surf_interp/', help='output directory')
    parser.add_argument('--mesh', dest='mesh_fn', default='/scratch/tfunck/output2/5_surf_interp/surfaces/surf_0.5mm_1.0_inflate_rsl.surf.gii', help='mesh fn')
    parser.add_argument('--ligand-csv', dest=file_list, nargs='+', default='/scratch/tfunck/output2/5_surf_interp/surfaces/surf_0.5mm_1.0_inflate_rsl.surf.gii', help='csv ligand densities')

    out_dir = args.out_dir
    mesh_fn = args.mesh_fn
    file_list = args.file_list

    #load raw data
    for filename in file_list :
        densities = pd.read_csv(filename,header=None).values.astype(np.float16)
        idx = densities > 0.0
        idx = idx.reshape(-1,)
        idx_range = np.arange(idx.shape[0]).astype(int)

        coords, faces = read_gifti(mesh_fn)
        print('Calculate neighbours')
        ngh = get_ngh(coords,faces)

        depth_list=range(1,11)
        for i, (x,y,z) in zip(idx_range[idx], coords[idx,:]): 
            print('i=',i)
            for max_depth in depth_list :
                # calculate neighbours
                core, border, dist = calculate_neighbours(i, ngh, coords, densities, max_depth)
                print('core',core)
                print('border',border)
                # interpolate small patch
                error = interpolate_over_sphere(densities, coords, ngh, core, border,i)
                print(error)
            exit(0)
                # calculate error


