import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import utils.ants_nibabel as nib
import nibabel as nb_surf
import pandas as pd
import numpy as np
import tempfile
import time
import re
import multiprocessing
import h5py as h5
from nibabel import freesurfer
from utils.utils import  transform_surface_to_slabs
from scipy.ndimage import label
from re import sub
from glob import glob
from utils.mesh_io import  load_mesh, load_mesh_geometry, save_mesh, save_mesh_data, save_obj, read_obj
from utils.utils import shell, w2v, v2w
from upsample_gifti import save_gii, upsample_gifti, identify_target_edges, obj_to_gii

global surf_base_str
surf_base_str = '{}/{}_{}_surface_{}_{}{}.{}'


class Ext():
    def __init__(self,gm_sphere,wm_sphere,gm,wm):
        self.gm_sphere = gm_sphere
        self.wm_sphere = wm_sphere
        self.gm = gm
        self.wm = wm


def get_surface_filename(surf_obj_fn, surf_gii_fn, surf_fs_fn):
    if os.path.exists(surf_gii_fn) :
        surf_fn = surf_gii_fn
        ext = Ext('.surf.gii', '.surf.gii', '.surf.gii', '.surf.gii')
    elif os.path.exists(surf_fs_fn):
        surf_fn = surf_fs_fn
        ext = Ext('.pial.sphere', '.white.sphere', '.pial', '.white')
    elif os.path.exists(surf_obj_fn):
        surf_fn = surf_gii_fn
        ext = Ext('.surf.gii', 'surf.gii', '.surf.gii', '.surf.gii')
        obj_to_gii(surf_obj_fn, ref_surf_fn, surf_fn)
    else :
        print('Error: could not find input GM surface (obj, fs, gii) for ', surf_gii_fn)
        exit(1)
    return surf_fn, ext


def prepare_surfaces(slab_dict, thickened_dict, depth_list, interp_dir, resolution, upsample_resolution, mni_fn, ligand, df_ligand, input_surf_dir='civet/mri1/surfaces/surfaces/', n_vertices = 327696, brain='mri1', hemi='R', clobber=0):
    '''

    '''
    surf_rsl_dir = interp_dir +'/surfaces/' 
    os.makedirs(surf_rsl_dir, exist_ok=True)
    
    #Interpolate at coordinate locations
    ref_surf_fn = surf_base_str.format( input_surf_dir, brain,'gray', hemi, n_vertices,'','surf.gii')
    ref_surf_obj_fn = surf_base_str.format( input_surf_dir, brain,'gray', hemi, n_vertices,'','obj')


    surf_gm_obj_fn = surf_base_str.format(input_surf_dir, brain, 'gray', hemi, n_vertices,'','obj')
    surf_wm_obj_fn = surf_base_str.format(input_surf_dir, brain, 'white', hemi, n_vertices,'','obj')
    
    surf_gm_gii_fn = surf_base_str.format(input_surf_dir, brain, 'gray', hemi, n_vertices,'','surf.gii')
    surf_wm_gii_fn = surf_base_str.format(input_surf_dir, brain, 'white', hemi, n_vertices,'','surf.gii')
   
    surf_gm_fs_fn = surf_base_str.format(input_surf_dir, brain, 'gray', hemi, n_vertices,'','pial')
    surf_wm_fs_fn = surf_base_str.format(input_surf_dir, brain, 'white', hemi, n_vertices,'','white')
    
    surf_gm_fn, ext = get_surface_filename(surf_gm_obj_fn, surf_gm_gii_fn, surf_gm_fs_fn)
    surf_wm_fn, _ = get_surface_filename(surf_wm_obj_fn, surf_wm_gii_fn, surf_wm_fs_fn)
    
    origin=[0,0,0]
    if ext.gm == '.pial' : origin= load_mesh(surf_gm_fn)[2]['cras']
    
    sphere_obj_fn = surf_base_str.format(input_surf_dir, brain, 'mid', hemi, n_vertices,'_sphere',ext.gm)
    
    #upsample transformed surfaces to given resolution
    surf_depth_mni_dict={}
    print("\tGenerating surfaces across cortical depth.") 
    surf_depth_mni_dict = generate_cortical_depth_surfaces(surf_depth_mni_dict, depth_list, resolution, surf_wm_fn, surf_gm_fn, surf_rsl_dir, ligand, ext)

    print("\tInflating surfaces.") 
    surf_depth_mni_dict = inflate_surfaces(surf_depth_mni_dict, surf_rsl_dir, ext, resolution,  depth_list,  ligand, clobber=clobber)

    print("\tUpsampling surfaces.") 
    surf_depth_mni_dict = upsample_surfaces(surf_depth_mni_dict, thickened_dict, surf_rsl_dir, surf_gm_fn, ext, resolution, upsample_resolution,  depth_list, slab_dict, ligand, df_ligand, clobber=clobber)

    #For each slab, transform the mesh surface to the receptor space
    print("\tTransforming surfaces to slab space.")
    surf_depth_slab_dict={}
    for slab in slab_dict.keys(): surf_depth_slab_dict[slab]={}
    
    for depth, depth_dict in surf_depth_mni_dict.items():
        surf_fn = depth_dict['upsample_fn']
        
        faces_fn = depth_dict['faces_fn']
        ref_gii_fn=depth_dict['depth_surf_fn']

        print(surf_fn)
        temp_dict={}
        temp_dict = transform_surface_to_slabs(temp_dict, slab_dict, thickened_dict,  surf_rsl_dir, surf_fn, faces_fn=faces_fn, ref_gii_fn=ref_gii_fn, ext='.surf.gii')

        for slab in slab_dict.keys() :
            surf_depth_slab_dict[slab][depth] = temp_dict[slab]
    
    return surf_depth_mni_dict, surf_depth_slab_dict, origin


def generate_cortical_depth_surfaces(surf_depth_mni_dict, depth_list, resolution, wm_surf_fn, surf_gm_fn, surf_dir, ligand, ext):

    gm_coords, gm_faces, gm_info = load_mesh(surf_gm_fn)
    wm_coords, wm_faces, wm_info = load_mesh(wm_surf_fn)

    if ext.wm == '.white' : volume_info=gm_info
    else : volume_info = surf_gm_fn

    d_coords = wm_coords - gm_coords 
    del wm_coords

    for depth in depth_list :
        depth_surf_fn = "{}/surf_{}_{}mm_{}{}".format(surf_dir,ligand,resolution,depth, ext.gm)
        surf_depth_mni_dict[depth]={'depth_surf_fn':depth_surf_fn}
        
        coords = gm_coords + depth * d_coords
        
        if not os.path.exists(depth_surf_fn) :
            save_mesh(depth_surf_fn, coords, gm_faces, volume_info=volume_info)
        del coords


    return surf_depth_mni_dict

def inflate_surfaces(surf_depth_mni_dict, surf_dir, ext, resolution,  depth_list, ligand,  clobber=False ):
    # Upsampling of meshes at various depths across cortex produces meshes with different n vertices.
    # To create a set of meshes across the surfaces across the cortex that have the same number of 
    # vertices, we first upsample and inflate the wm mesh.
    # Then, for each mesh across the cortex we resample that mesh so that it has the same polygons
    # and hence the same number of vertices as the wm mesh.
    # Each mesh across the cortical depth is inflated (from low resolution, not the upsampled version)
    # and then resampled so that it has the high resolution number of vertices.

    gm_sphere_fn = "{}/surf_{}mm_{}{}".format(surf_dir, resolution,0, ext.gm_sphere)
    gm_sphere_rsl_fn = "{}/surf_{}mm_{}rsl{}".format(surf_dir, resolution,0, ext.gm_sphere)
    
    
    for depth in depth_list :
        depth += 0.
        print("\tDepth", depth)
        print(surf_depth_mni_dict[float(depth)])
        depth_surf_fn = surf_depth_mni_dict[float(depth)]['depth_surf_fn']
        sphere_fn = "{}/surf_{}_{}mm_{}_inflate{}".format(surf_dir,ligand,resolution,depth, ext.gm_sphere)
        sphere_rsl_fn = "{}/surf_{}_{}mm_{}_inflate_rsl.h5".format(surf_dir,ligand,resolution,depth)
        surf_depth_mni_dict[depth]['sphere_rsl_fn']=sphere_rsl_fn
        surf_depth_mni_dict[depth]['sphere_fn']=sphere_fn
        
        if not os.path.exists(sphere_fn) or  clobber :
            print('\tInflate to sphere')
            #shell('~/freesurfer/bin/mris_inflate -n 500  {} {}'.format(depth_surf_fn, sphere_fn))
            shell('~/freesurfer/bin/mris_inflate -n 25  {} {}'.format(depth_surf_fn, sphere_fn))

    return surf_depth_mni_dict

def upsample_surfaces(surf_depth_mni_dict, thickened_dict, surf_dir, surf_gm_fn, ext, resolution, upsample_resolution,  depth_list, slab_dict, ligand, df_ligand, clobber=False):

    gm_obj_fn="{}/surf_{}_{}mm_{}_rsl.obj".format(surf_dir,ligand, resolution,0)
    upsample_0_fn = "{}/surf_{}_{}mm_{}_rsl{}".format(surf_dir, ligand, resolution,depth_list[0], ext.gm)
    upsample_1_fn = "{}/surf_{}_{}mm_{}_rsl{}".format(surf_dir, ligand, resolution,depth_list[-1], ext.gm)
    input_list = []
    output_list = []

    surf_slab_space_dict={}
    for depth in depth_list :

        upsample_gii_fn = "{}/surf_{}_{}mm_{}_rsl{}".format(surf_dir,ligand, resolution,depth, ext.gm)
        upsample_fn = "{}/surf_{}_{}mm_{}_rsl.h5".format(surf_dir,ligand, resolution,depth)

        surf_depth_mni_dict[depth]['upsample_fn']=upsample_fn
        surf_depth_mni_dict[depth]['upsample_gii_fn']=upsample_gii_fn
        
        depth_surf_fn = surf_depth_mni_dict[depth]['depth_surf_fn']
        sphere_fn = surf_depth_mni_dict[depth]['sphere_fn']
        sphere_rsl_fn = surf_depth_mni_dict[depth]['sphere_rsl_fn']

        input_list += [depth_surf_fn, sphere_fn]
        output_list+= [upsample_fn, sphere_rsl_fn]

        surf_slab_space_dict = transform_surface_to_slabs(surf_slab_space_dict, slab_dict, thickened_dict, surf_dir, surf_gm_fn, ext='.surf.gii')

    surf_depth_mni_dict[depth_list[0]]['upsample_gii_fn'] = upsample_0_fn
    surf_depth_mni_dict[depth_list[-1]]['upsample_gii_fn'] = upsample_1_fn
     
    faces_fn, coords_fn = upsample_gifti(surf_gm_fn, upsample_0_fn, upsample_1_fn, float(upsample_resolution), df_ligand, input_list=input_list, output_list=output_list, surf_slab_space_dict=surf_slab_space_dict, clobber=clobber)
   
    for depth in surf_depth_mni_dict.keys():
        surf_depth_mni_dict[depth]['faces_fn'] = faces_fn
    
    rsl_faces = h5.File(faces_fn,'r')['data'][:]
    rsl_coords = h5.File(coords_fn, 'r')['data'][:]
    if not os.path.exists(gm_obj_fn) :
        save_obj(gm_obj_fn, rsl_coords,rsl_faces)

    return surf_depth_mni_dict

