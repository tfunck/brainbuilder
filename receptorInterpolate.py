import numpy as np
import nibabel as nib
import pandas as pd
import sys
import os
import json
import re
import matplotlib.pyplot as plt
import pandas as pd
import time
import stripy as stripy
#import psutil
import vast.surface_tools as surface_tools
import vast.surface_tools as io_mesh
from glob import glob
from ants import  apply_transforms, from_numpy
from ants import  from_numpy
from ANTs import ANTs
from ants import image_read, registration
from utils.utils import splitext, shell


def alignLigandToSRV(df, slab , ligand, srv_fn, cls_fn, output_dir,  tfm_type_2d="SyNAggro", clobber=False):
    print("\t\tNonlinear alignment of coronal sections for ligand",ligand,"for slab",int(slab))
    srv=None
    cls=None
    tfm={}
    tfm_dir=output_dir + os.sep + 'tfm'
    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)
    
    print("srv_fn:", srv_fn)
    srv_img = nib.load(srv_fn)
    srv = srv_img.get_data()
    
    cls = nib.load(cls_fn).get_data()
    
    for i, row in df.iterrows() :
        _y0 = row["volume_order"]
        print(_y0)
        tfm[str(_y0)]={str(_y0):[]}

        prefix_string= ''.join(["slab-",str(slab),"_ligand-",ligand,"_y-",str(_y0)])
        prefix=tfm_dir+os.sep + ''.join(["slab-",str(slab),"_ligand-",ligand,"_y-",str(_y0), "_"]) + os.sep 
        if not os.path.exists(prefix) : 
            os.makedirs(prefix)
        
        print(not os.path.exists(prefix+"1Warp.nii.gz"), prefix+"1Warp.nii.gz" , clobber)
        if not os.path.exists(prefix+"1Warp.nii.gz") or clobber : #not os.path.exists(prefix+'DenseRigid_0GenericAffine.mat') or clobber :
            srv_slice = from_numpy( srv[ :, int(_y0), : ] )
            cls_slice = from_numpy( cls[ :, int(_y0), : ] )
            print('SyN')
            reg = registration(fixed=srv_slice, moving=cls_slice, type_of_transform=tfm_type_2d,reg_iterations=(1000,750,500), aff_metric='GC', syn_metic='CC', outprefix=prefix  )
            print(reg)
            tfm[str(_y0)][str(_y0)] = reg['fwdtransforms'] +tfm[str(_y0)][str(_y0)]
        else :
            tfm[str(_y0)][str(_y0)] = [prefix+'1Warp.nii.gz', prefix+'0GenericAffine.mat' ] + tfm[str(_y0)][str(_y0)]

    return tfm


def receptorInterpolate( slab, out_fn, rec_fn, srv_fn, cls_fn, output_dir, ligand, slice_info_fn,  subslab=-1, tfm_type_2d='SyNAggro', clobber=False , validation=False) :
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
    print("Subslab", subslab) 
    #Create name for json file that stores transformations to map raw/linearized 2D autoradiographs
    #into alignment with super-resolution GM representation extracted from MRI
    tfm_dict_fn = output_dir + os.sep + ligand +"_slab-"+str(slab)+"_tfm.json" 
    ligand_out_fn = output_dir + os.sep + ligand +"_slab-"+str(slab)+"_init.nii.gz" 
    
    # 1. Get locations of slices for particular ligand in specified slab
    df = pd.read_csv(slice_info_fn)
    if ligand != 'all':
        df = df.loc[ df["ligand"] == ligand ]
    if subslab == None : subslab=-1
    if subslab >= 0  :
        print('Running sub-slab: ', subslab, '/', df.shape[0]-1)
        df = df.iloc[ subslab:(subslab+2), :]

    # 2. Find non-linear transform from autoradiograph GM classified slice to MRI-derived SRV image
    tfm = alignLigandToSRV(df, slab, ligand, srv_fn, cls_fn, output_dir,  tfm_type_2d=tfm_type_2d, clobber=clobber)

    surf_dict = load_mesh_geometry(surf_obj_fn)
    coords = surf_dict['coords']

    print('Get surface mask and surface values')
    surface_mask_fn = f'surface_slab_mask_{n_vertices}.txt'
    surface_val_fn  = f'surface_slab_val_{n_vertices}.txt'
    surface_mask, surface_val = get_surface_values(mask_src, array_src, coords, surface_mask_fn, surface_val_fn)


    print('Min',np.min(surface_val),'Max',np.max(surface_val))
    #load in sphere mesh
    print('Load mesh sphere')
    sphere = load_mesh_geometry(sphere_obj_fn) 

    print(surface_mask.shape)
    print('Convert to sphereical coordinates')
    spherical_coords = surface_tools.spherical_np(sphere['coords'])
    print(spherical_coords.shape)

    spherical_coords_src = spherical_coords[ surface_mask, : ]

    surface_val_src = surface_val[ surface_mask.astype(bool) ]
    surface_val_tgt = surface_val[ ~surface_mask.astype(bool) ]

    lats_src, lons_src = spherical_coords_src[:,1]-np.pi/2, spherical_coords_src[:,2]
    lats, lons = spherical_coords[:,1]-np.pi/2, spherical_coords[:,2]

    print('Create Mesh')
    mesh = stripy.sTriangulation(lons_src,lats_src)

    print('Interpolate')
    interpolated_data=np.zeros_like(surface_val)
    interpolated_data[surface_mask]=surface_val_src
    interp_val, interp_type = mesh.interpolate(lons,lats, zdata=surface_val[surface_mask], order=1)
    print(interp_val.shape, lats.shape)
    interpolated_data=interp_val
    print('Write interpolated values')
    pd.DataFrame(interpolated_data).to_csv('surface_interpolated_values.txt',index=False)
   
    
