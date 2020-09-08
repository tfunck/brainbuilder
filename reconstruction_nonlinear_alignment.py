import os
import json
import numpy as np
import nibabel as nib
import argparse
from ANTs import ANTs
from nibabel.processing import resample_from_to
from utils.utils import shell, downsample_and_crop
from sys import exit
from re import sub
from kmeans_vol import kmeans_vol
from glob import glob
from receptorCrop import crop_source_files

def ants_transform_surface(ii_fn, tfm_fn, out_fn) :

    surf_dict = io_mesh.load_mesh_geometry(ii_fn)
    coords = surf_dict['coords']

    surf_dict['coords'] = ants.apply_ants_transform_to_vector( ants.read_transform(tfm_fn) , coords)

    save_mesh_data(out_fn, surf_data)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--brains','-b', dest='brain', type=str, help='Brains to reconstruct. Default = run all.')
    parser.add_argument('--hemispheres', '--hemi', dest='hemi',type=str, help='Brains to reconstruct. Default = reconstruct all hemispheres.')
    parser.add_argument('--slab','-s', dest='slab', type=str, help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--fixed','-f', dest='fixed_fn', type=str, help='')
    parser.add_argument('--moving','-m', dest='moving_fn', type=str, help='')
    parser.add_argument('--out','-o', dest='out_fn', type=str, help='')
    parser.add_argument('--init-tfm','-t', dest='init_tfm', type=str, help='')
    parser.add_argument('--shrink-factors', dest='shrink_factors', type=str, default='1')
    parser.add_argument('--smooth-factors', dest='smooth_sigmas', type=str, default='0')
    parser.add_argument('--iterations', dest='iterations', type=str, default='1000')
    parser.add_argument('--df-fn', dest='df_fn', type=str, help='')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')
    
    args = parser.parse_args()

    brain = args.brain
    hemi = args.hemi
    slab = args.slab
    clobber = args.clobber


    iterations=[ args.iterations ]
    shrink_factors=[ args.shrink_factors]
    smoothing_sigmas=[ args.smooth_sigmas ]
    metrics=[ 'Mattes' ]
    tfm_types=['SyN']
    
    prefix = os.path.splitext(args.out_fn)[0]
    out_dir = os.path.dirname(prefix) + os.sep

    if not os.path.exists(out_dir) :
        os.makedirs(out_dir)

    #shell(' '.join(['antsApplyTransforms -v 1 -i', args.moving_fn, '-r', args.fixed_fn, '-o', 'test_lin.nii.gz', '-t [', args.init_tfm,',1]']))
    #tfm_syn, moving_rsl_fn = ANTs(prefix+'_', args.fixed_fn, 'test_lin.nii.gz', prefix, iterations=iterations, tfm_type=tfm_types, shrink_factors=shrink_factors, sampling=1, rate=[0.5], metrics=['Mattes'], smoothing_sigmas=smoothing_sigmas, radius=3,  verbose=1, clobber=0,  generate_masks=False, no_init_tfm=False, init_inverse=True, exit_on_failure=True)

    #if ( not os.path.exists(args.out_fn) or clobber ) and moving_rsl_fn != None :
    #shell(' '.join(['antsApplyTransforms -v 1 -i', 'test_lin.nii.gz', '-r', args.fixed_fn, '-o', 'test_nl.nii.gz', '-t', tfm_syn ]))
    #shell(' '.join(['antsApplyTransforms -v 1 -i', args.moving_fn, '-r', args.fixed_fn, '-o', 'test_nl.nii.gz', '-t', tfm_syn ]))

