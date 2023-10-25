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
from utils.mesh_utils import  transform_surface_to_slabs,  upsample_over_faces,   mesh_to_volume, load_mesh_ext, get_triangle_vectors, unique_points

from scipy.ndimage import label
from re import sub
from glob import glob
from utils.mesh_io import  load_mesh, load_mesh_geometry, save_mesh, save_mesh_data, save_obj, read_obj
from utils.utils import shell, w2v, v2w

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



def get_gm_wm_surfaces(input_surf_dir, brain, hemi, n_vertices):
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

    return surf_gm_fn, surf_wm_fn, ext

def prepare_surfaces(slab_dict, ligandSlabData,  mni_fn, df_ligand, input_surf_dir='civet/mri1/surfaces/surfaces/', n_vertices = 327696, brain='mri1', hemi='R', clobber=0):
    '''

    '''
    depth_list=ligandSlabData.depths
    interp_dir = ligandSlabData.volume_dir
    resolution = ligandSlabData.resolution
    ligand=ligandSlabData.ligand
    brain=ligandSlabData.brain
    hemi=ligandSlabData.hemi

    surf_rsl_dir = ligandSlabData.surface_dir 
    os.makedirs(surf_rsl_dir, exist_ok=True)
    
    #Interpolate at coordinate locations
    surf_gm_fn, surf_wm_fn, ext = get_gm_wm_surfaces(input_surf_dir, brain, hemi, n_vertices)
    
    origin=[0,0,0]
    #DEBUG if ext.gm == '.pial' : origin = load_mesh(surf_gm_fn)[2]['cras']
    print(surf_gm_fn)
    coords = load_mesh(surf_gm_fn, correct_offset=True)[0]
   
    img = nb_surf.load(mni_fn)
    dimensions = img.shape
    steps = img.affine[[0,1,2],[0,1,2]]
    starts = img.affine[[0,1,2],[3,3,3]]
    print(steps)
    print(starts)
    interp_vol, _  = mesh_to_volume(coords, np.ones(coords.shape[0]), dimensions, starts, steps)
    nib.Nifti1Image(interp_vol, nib.load(mni_fn).affine,direction_order='lpi').to_filename(f'{interp_dir}/tmp.nii.gz')

    sphere_obj_fn = surf_base_str.format(input_surf_dir, brain, 'mid', hemi, n_vertices,'_sphere',ext.gm)
    print('\tSurf GM filename:', surf_gm_fn) 
    #upsample transformed surfaces to given resolution
    surf_depth_mni_dict={}

    print("\tGenerating surfaces across cortical depth.") 
    surf_depth_mni_dict = generate_cortical_depth_surfaces(surf_depth_mni_dict, mni_fn, depth_list, resolution, surf_wm_fn, surf_gm_fn, surf_rsl_dir, ligand, ext)

    print("\tInflating surfaces.") 
    surf_depth_mni_dict = inflate_surfaces(surf_depth_mni_dict, surf_rsl_dir, ext, resolution,  depth_list,  clobber=clobber)

    print("\tUpsampling surfaces.") 
    surf_depth_mni_dict = upsample_surfaces(surf_depth_mni_dict, ligandSlabData.volumes, surf_rsl_dir, surf_gm_fn, ext, resolution, depth_list, slab_dict, ligand, df_ligand, mni_fn, clobber=clobber)

    #For each slab, transform the mesh surface to the receptor space
    print("\tTransforming surfaces to slab space.")
    transfrom_depth_surf_to_slab_space(slab_dict, surf_depth_mni_dict, ligandSlabData.volumes, surf_rsl_dir)

    return surf_depth_mni_dict, origin



def transfrom_depth_surf_to_slab_space(slab_dict, surf_depth_mni_dict, volumes, surf_rsl_dir):

    
    for depth, depth_dict in surf_depth_mni_dict.items():
        faces_fn=None

        ref_gii_fn=depth_dict['depth_surf_fn']

        surf_fn = depth_dict['depth_rsl_fn']
        transform_surface_to_slabs( volumes, depth, slab_dict,  surf_rsl_dir, surf_fn, faces_fn=faces_fn, ref_gii_fn=ref_gii_fn, ext='.surf.gii')



def generate_cortical_depth_surfaces(surf_depth_mni_dict, ref_vol_fn, depth_list, resolution, wm_surf_fn, surf_gm_fn, surf_dir, ligand, ext):

    img = nb_surf.load(ref_vol_fn)
    dimensions = img.shape
    steps = img.affine[[0,1,2],[0,1,2]]
    starts = img.affine[[0,1,2],[3,3,3]]

    gm_coords, gm_faces, gm_info = load_mesh(surf_gm_fn, correct_offset=True)
    wm_coords, wm_faces, wm_info = load_mesh(wm_surf_fn, correct_offset=True)

    #if gm_info != None:
    #    print('Volume info (origin):', gm_info['cras'])
    #    gm_coords += gm_info['cras']

    #if wm_info != None:
    #    print('Volume info (origin):', wm_info['cras'])
    #    wm_coords += wm_info['cras']


    if ext.wm == '.white' : volume_info=gm_info
    else : volume_info = surf_gm_fn

    d_coords = wm_coords - gm_coords 
    del wm_coords

    for depth in depth_list :
        depth_surf_fn = "{}/surf_{}mm_{}{}".format(surf_dir,resolution,depth, ext.gm)
        depth_vol_fn = "{}/surf_{}mm_{}{}".format(surf_dir,resolution,depth, '.nii.gz')
        surf_depth_mni_dict[depth]={'depth_surf_fn':depth_surf_fn}
        
        coords = gm_coords + depth * d_coords
        
        if not os.path.exists(depth_surf_fn) :
            save_mesh(depth_surf_fn, coords, gm_faces, volume_info=volume_info)
            interp_vol, _  = mesh_to_volume(coords, np.ones(coords.shape[0]), dimensions, starts, steps)
            print('\tWriting', depth_vol_fn)
            nib.Nifti1Image(interp_vol, nib.load(ref_vol_fn).affine,direction_order='lpi').to_filename(depth_vol_fn)


    return surf_depth_mni_dict

def inflate_surfaces(surf_depth_mni_dict, surf_dir, ext, resolution,  depth_list,  clobber=False ):
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
        inflate_fn = "{}/surf_{}mm_{}.inflate".format(surf_dir,resolution,depth)
        sphere_fn = "{}/surf_{}mm_{}.sphere".format(surf_dir,resolution,depth)
        sphere_rsl_fn = "{}/surf_{}mm_{}_sphere_rsl.npz".format(surf_dir,resolution,depth)
        surf_depth_mni_dict[depth]['sphere_rsl_fn']=sphere_rsl_fn
        surf_depth_mni_dict[depth]['sphere_fn']=sphere_fn
        surf_depth_mni_dict[depth]['inflate_fn']=inflate_fn
        
        if not os.path.exists(sphere_fn) or  clobber :
            print('\tInflate to sphere')
            shell(f'~/freesurfer/bin/mris_inflate -n 100  {depth_surf_fn} {inflate_fn}')
            print(f'\t\t{inflate_fn}')
            shell(f'~/freesurfer/bin/mris_sphere -q  {inflate_fn} {sphere_fn}')
            print(f'\t\t{sphere_fn}')

    return surf_depth_mni_dict

def generate_face_and_coord_mask(edge_mask_idx, faces, coords):
    mask_idx_unique = np.unique(edge_mask_idx)
    n = max(faces.shape[0], coords.shape[0])

    face_mask=np.zeros(faces.shape[0]).astype(np.bool)
    coord_mask=np.zeros(coords.shape[0]).astype(np.bool)

    for i in range(n) :
        if i < faces.shape[0] :
            f0,f1,f2 = faces[i]
            if f0 in mask_idx_unique : face_mask[i]=True
            if f1 in mask_idx_unique : face_mask[i]=True
            if f2 in mask_idx_unique : face_mask[i]=True
        if i < coords.shape[0]:
            if i in mask_idx_unique :
                coord_mask[i]=True
    return face_mask, coord_mask



def resample_points(surf_fn, new_points_gen):
    points, faces = load_mesh_ext(surf_fn)
    n = len(new_points_gen)

    new_points = np.zeros([n,3])

    for i, gen in enumerate(new_points_gen) :
        new_points[i] = gen.generate_point(points)

    #new_points, unique_index, unique_reverse = unique_points(new_points)
    new_points += np.random.normal(0,0.0001, new_points.shape )
    return new_points, points

def upsample_surfaces(surf_depth_mni_dict, thickened_dict, surf_dir, surf_gm_fn, ext, resolution, depth_list, slab_dict, ligand, df_ligand, mni_fn, clobber=False):

    gm_obj_fn="{}/surf_{}mm_{}_rsl.obj".format(surf_dir, resolution,0)
    upsample_0_fn = "{}/surf_{}mm_{}_rsl{}".format(surf_dir, resolution,depth_list[0], ext.gm)
    upsample_1_fn = "{}/surf_{}mm_{}_rsl{}".format(surf_dir, resolution,depth_list[-1], ext.gm)
    input_list = []
    output_list = []

    surf_slab_space_dict={}
    for depth in depth_list :

        depth_rsl_gii = "{}/surf_{}mm_{}_rsl{}".format(surf_dir, resolution,depth, ext.gm)
        depth_rsl_fn = "{}/surf_{}mm_{}_rsl.npz".format(surf_dir, resolution,depth)

        surf_depth_mni_dict[depth]['depth_rsl_fn']=depth_rsl_fn
        surf_depth_mni_dict[depth]['depth_rsl_gii']=depth_rsl_gii
        
        depth_surf_fn = surf_depth_mni_dict[depth]['depth_surf_fn']
        sphere_fn = surf_depth_mni_dict[depth]['sphere_fn']
        sphere_rsl_fn = surf_depth_mni_dict[depth]['sphere_rsl_fn']

        input_list += [depth_surf_fn, sphere_fn]
        output_list+= [depth_rsl_fn, sphere_rsl_fn]

        

    surf_depth_mni_dict[depth_list[0]]['depth_rsl_gii'] = upsample_0_fn
    surf_depth_mni_dict[depth_list[-1]]['depth_rsl_gii'] = upsample_1_fn
    
    #DEBUG put the next few lines incase want to try face upsampling instead of full surf upsampling
    ref_gii_fn = surf_depth_mni_dict[depth_list[0]]['depth_surf_fn']
    ref_rsl_gii_fn = surf_depth_mni_dict[depth_list[0]]['depth_rsl_gii']
    ref_rsl_npy_fn = sub('.surf.gii', '', surf_depth_mni_dict[depth_list[0]]['depth_rsl_gii'])
    ngh_npz_fn = sub('.nii.gz', '_ngh', surf_depth_mni_dict[depth_list[0]]['depth_rsl_gii'])
    coords, faces, volume_info = load_mesh(ref_gii_fn)
    

    if False in [ os.path.exists(fn) for fn in output_list+[ref_rsl_npy_fn+'.npz']]:
        points, _, new_points_gen  = upsample_over_faces(ref_gii_fn, resolution, ref_rsl_npy_fn)
    
        img = nb_surf.load(mni_fn)
        #img = nib.load(mni_fn)
        steps=img.affine[[0,1,2],[0,1,2]]
        starts=img.affine[[0,1,2],3]
        dimensions=img.shape
        #points, _ = load_mesh_ext(ref_gii_fn)
        interp_vol, _  = mesh_to_volume(points, np.ones(points.shape[0]), dimensions, starts, steps)
        nib.Nifti1Image(interp_vol, nib.load(mni_fn).affine,direction_order='lpi').to_filename(f'{surf_dir}/surf_{resolution}mm_{depth}_rsl.nii.gz')
        print('hello')
        print(f'{surf_dir}/surf_{resolution}mm_{depth}_rsl.nii.gz')
        
        print('Upsampled points', points.shape, len(new_points_gen)) 
        #DEBUG
        #test_points, old_points = resample_points(ref_gii_fn, new_points_gen)
        #assert test_points.shape[0] == points.shape[0], 'Error, mismatch when testing resampling'

        ref_points = np.load(ref_rsl_npy_fn+'.npz')['points']
        n_points = ref_points.shape[0]
        
        for in_fn, out_fn in zip(input_list, output_list):
            points, old_points = resample_points(in_fn, new_points_gen)
            #full = np.arange(points.shape[0]).astype(int)

            assert n_points == points.shape[0], f'Error: mismatch in number of points in mesh between {n_points} and {points.shape[0]}'
            interp_vol, _  = mesh_to_volume(points, np.ones(points.shape[0]), dimensions, starts, steps)
            nii_fn = os.path.splitext(out_fn)[0]+'.nii.gz'
            nib.Nifti1Image(interp_vol, nib.load(mni_fn).affine,direction_order='lpi').to_filename(nii_fn)
            print('\tWriting (nifti):', nii_fn)
            assert points.shape[1], 'Error: shape of points is incorrect ' + points.shape 
            np.savez(out_fn, points=points)
            print('\tWriting (npz)', out_fn)
    #np.savez(ngh_npz_fn, ngh=ngh)
    #ngh = np.load(ngh_npz_fn+'.npz')['ngh']
    
    #extra_points = coords[ ~ coord_mask,:]
    #points = np.concatenate([extra_points, points],axis=0)
    #print('\t\tUsing:', ngh_npz_fn+'.npz')
    #if not os.path.exists(ngh_npz_fn+'.npz') or True:
    #    ngh = link_points(points, ngh, resolution)
    #faces = get_faces_from_neighbours(ngh)
    #save_mesh(out_fn, points, faces, surf_fn)
    #faces_fn, coords_fn = upsample_gifti(surf_gm_fn, upsample_0_fn, upsample_1_fn, float(upsample_resolution), df_ligand, input_list=input_list, output_list=output_list, surf_slab_space_dict=surf_slab_space_dict, clobber=clobber)
   
    #for depth in surf_depth_mni_dict.keys():
    #    surf_depth_mni_dict[depth]['faces_fn'] = faces_fn
    
    #rsl_faces = h5.File(faces_fn,'r')['data'][:]
    #rsl_coords = h5.File(coords_fn, 'r')['data'][:]
    #if not os.path.exists(gm_obj_fn) :
    #    save_obj(gm_obj_fn, rsl_coords,rsl_faces)
    return surf_depth_mni_dict

