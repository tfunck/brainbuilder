from utils.utils import shell, resample
from utils.ANTs import ANTs
from nibabel.processing import resample_to_output
from reconstruction.align_slab_to_mri import align_slab_to_mri
from reconstruction.receptor_segment import classifyReceptorSlices
from reconstruction.nonlinear_2d_alignment import receptor_2d_alignment
from reconstruction.init_alignment import receptorRegister
from reconstruction.surface_interpolation import surface_interpolation
from reconstruction.crop import crop
import pandas as pd
import numpy as np
import nibabel as nib
import os
import json
import re
import argparse

global file_dir
base_file_dir, fn =os.path.split( os.path.abspath(__file__) )
file_dir = base_file_dir +os.sep +'section_numbers' +os.sep
def w2v(c, step, start):
    return np.round( (c-start)/step ).astype(int)

def adjust_slab_list(ll) :
    ll = [int(i) for i in ll ]
    ll.sort()
    oo=[]
    for i in range(len(ll)):
        j = int(i/2)
        if i % 2 :
            oo.append(ll[ -(j+1)]  )
        else :
            oo.append(ll[j])
    oo = [ str(i) for i in oo ]
    return oo

def calculate_section_order(autoradiograph_info_fn, source_dir, out_dir, in_df_fn='section_order/autoradiograph_info.csv') :
    df=pd.read_csv(in_df_fn)
    df['volume_order']=-1
    for (brain,hemi), tdf in df.groupby(['mri', 'hemisphere']):
        tdf = tdf.sort_values(['slab'])
        prev_max = 0
        for slab, sdf in tdf.groupby(['slab']):
            idx=(df['mri']==brain) & (df['hemisphere']==hemi) & (df['slab']==slab)
            df['global_order'].loc[ idx ] = sdf['slab_order'] + prev_max
            df['volume_order'].loc[ idx ] = sdf['slab_order'].max() - sdf['slab_order']
            prev_max =  sdf['slab_order'].max() + prev_max
            print(brain,hemi,slab, df['volume_order'].loc[idx].min(), df['volume_order'].loc[idx].max(),df['global_order'].loc[idx].min(), df['global_order'].loc[idx].max(), prev_max)

    df['filename']=source_dir+'/'+df['lin_fn'].apply(lambda x : os.path.splitext(os.path.basename(x))[0]) +'.png'
    df['filename']=df['filename'].apply(lambda x : re.sub('#L.png','.png',x))
    df['filename_rsl'] = df['filename'].apply(lambda x: out_dir+'/reconstruction_output/0_crop/'+os.path.splitext(os.path.basename(x))[0]+'#L.nii.gz')

    df.sort_values(["mri","hemisphere","slab","volume_order"], inplace=True)

    # Remove UB , "unspecific binding" from the df
    df = df.loc[df['repeat'] != 'UB']

    df.to_csv(autoradiograph_info_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--brains','-b', dest='brain', type=str, default='MR1', help='Brains to reconstruct. Default = run all.')
    parser.add_argument('--hemispheres', '--hemi', dest='hemi',default='R',type=str, help='Brains to reconstruct. Default = reconstruct all hemispheres.')
    parser.add_argument('--clobber', , dest='clobber',default=False,action='store_true', help='Overwrite existing results')
    parser.add_argument('--slab','-s', dest='slab', type=str, default=1, help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--src-dir','-i', dest='src_dir', type=str, default='receptor_dwn', help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--out-dir','-o', dest='out_dir', type=str, default='reconstruction_output', help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--remote','-p', dest='remote', default=False, action='store_true',  help='Slabs to reconstruct. Default = reconstruct all slabs.')

    args = parser.parse_args()
    ###
    ### Parameters
    ###
    brain_list=[args.brain] 
    hemi_list=[args.hemi]
    slab_list=[args.slab] 
    resolution_list = [ 3, 2, 1 , 0.5, 0.25]

    slab_list = adjust_slab_list(slab_list)

    ### If running on server:
    #run=qsub
    ### If running locally:
    run='sh' 
    scale_factors_fn=base_file_dir+'/scale_factors.json'
    srv_fn="srv/mri1_gm_bg_srv.nii.gz"
    srv_base_fn = srv_fn

    out_dir=args.out_dir #'reconstruction_output'
    src_dir=args.src_dir #'receptor_dwn/'
    crop_dir=f'{out_dir}/0_crop'
    os.makedirs(out_dir,exist_ok=True)

    autoradiograph_info_fn=out_dir+'/autoradiograph_info_volume_order.csv'

    #Process the base autoradiograph csv
    if not os.path.exists(autoradiograph_info_fn) : 
        calculate_section_order(autoradiograph_info_fn, crop_dir, out_dir, in_df_fn=file_dir+os.sep+'autoradiograph_info.csv')

    df=pd.read_csv(autoradiograph_info_fn)

    files={}
    ###---------------------###
    ###  PROCESSING STEPS   ###
    ###---------------------###
    #   0. Crop all autoradiographs
    #   1. Init Alignment (Rigid 2D, per slab)
    #   2. GM segmentation of receptor volumes (per slab)
    #   3. Slab to MRI (Affine 3D, per slab)
    #   4. GM MRI to autoradiograph volume (Nonlinear 3D, per slab)
    #   5. Autoradiograph to GM MRI (2D nonlinear, per slab)
    #   6. Interpolate autoradiograph to surface
    #   7. Interpolate missing vertices on sphere, interpolate back to 3D volume

    ### Step 0 :
    df = crop(src_dir,crop_dir, df,out_dir+os.sep+'autoradiograph_info_downsample.csv', remote=args.remote)

    ###
    ### Steps 1 & 2: Initial interslab alignment, GM segmentation
    ###
    #Paramaters
    n_epochs=10

    for brain in brain_list :
        files[brain]={}
        for hemi in hemi_list :
            files[brain][hemi]={}
            for slab in slab_list : 
                files[brain][hemi][slab]={}
                for resolution in resolution_list :
                    files[brain][hemi][slab][resolution] = {}

    for brain in brain_list :
        for hemi in hemi_list : 
            for slab_i, slab in enumerate(slab_list) : 
                ###
                ###  Step 1: Initial Alignment
                ###
                print('\tInitial rigid inter-autoradiograph alignment')
                out_dir_1=f'{out_dir}/{brain}_{hemi}_{slab}/1_init_align/'
                init_align_fn=f'{out_dir_1}/brain-{brain}_hemi-{hemi}_slab-{slab}_init_align.nii.gz'

                files[brain][hemi][slab][resolution_list[0]]['init_dir'] = out_dir_1
                files[brain][hemi][slab][resolution_list[0]]['init_align_fn'] = init_align_fn
                    
                slab_df=df.loc[  (df['hemisphere']==hemi) & (df['mri']==brain) & (df['slab']==int(slab)) ]

                if (not os.path.exists( init_align_fn) or args.clobber) and not args.remote : 
                    receptorRegister(brain,hemi,slab, init_align_fn, out_dir_1, slab_df, scale_factors_json=scale_factors_fn, clobber=args.clobber)
               

                ### Iterate over progressively finer resolution
                for resolution_itr, resolution in enumerate(resolution_list) :
                    print('Resolution',resolution)

                    cur_out_dir=f'{out_dir}/{brain}_{hemi}_{slab}/{resolution}mm/'
                    os.makedirs(cur_out_dir,exist_ok=True)
                    lin_dir=f'{cur_out_dir}/3_align_slab_to_mri/'
                    files[brain][hemi][slab]['lin_dir'] = lin_dir

                    ###
                    ### Step 1.5 : Downsample SRV to current resolution
                    ###
                    srv_rsl_fn = f'{cur_out_dir}/srv_{resolution}mm.nii.gz' 
                    if not os.path.exists(srv_rsl_fn) or args.clobber :
                        resample(nib.load(srv_fn), srv_rsl_fn, resolution)
     
                    srv_base_rsl_fn = f'{cur_out_dir}/srv_base_{resolution}mm.nii.gz' 
                    if not os.path.exists(srv_base_rsl_fn) or args.clobber :
                        resample(nib.load(srv_base_fn), srv_base_rsl_fn, resolution)

                    ###
                    ### Step 2 : Autoradiograph segmentation
                    ###
                    print('\tAutoradiograph segmentation')
                    if resolution_itr > 0 : align_fn = nl_2d_vol
                    else : align_fn = init_align_fn

                    seg_dir=f'{cur_out_dir}/2_segment/'
                    seg_rsl_fn=f'{seg_dir}/brain-{brain}_hemi-{hemi}_slab-{slab}_seg_{resolution}mm.nii.gz'
                    print(seg_rsl_fn, os.path.exists(seg_rsl_fn))
                    if not os.path.exists(seg_rsl_fn) or args.clobber:
                        classifyReceptorSlices(align_fn, seg_dir, seg_rsl_fn, rsl_dim=resolution)
                    
                    ###
                    ### Step 3 : Align slabs to MRI
                    ###
                    print('\tStep 3: align slabs to MRI')
                    hemi_df = df.loc[ (df['mri']==brain) & (df['hemisphere']==hemi) ]

                    align_to_mri_dir = f'{cur_out_dir}/3_align_slab_to_mri/' 
                    os.makedirs(align_to_mri_dir, exist_ok=True)
                    prev_resolution=resolution_list[resolution_itr-1]

                    if resolution_itr == 0:
                        tfm=None
                    else :
                        tfm=files[brain][hemi][slab][prev_resolution]['nl_3d_tfm']
                    
                    nl_3d_tfm_fn, nl_3d_tfm_inv_fn, rec_3d_lin, srv_3d_lin = align_slab_to_mri(seg_rsl_fn, srv_rsl_fn, slab, align_to_mri_dir, hemi_df, slab_list[slab_i:], tfm )

                    files[brain][hemi][slab][resolution]['nl_3d_tfm'] = nl_3d_tfm_fn
                    files[brain][hemi][slab][resolution]['nl_3d_tfm_inv'] = nl_3d_tfm_inv_fn

                    ###
                    ### Step 4 : 2D alignment of receptor to resample MRI GM vol
                    ###
                    nl_2d_dir= f'{cur_out_dir}/4_nonlinear_2d'
                    os.makedirs(nl_2d_dir, exist_ok=True)
                    files[brain][hemi][slab][resolution]['nl_2d_vol']=f"{nl_2d_dir}/{brain}_{hemi}_{slab}_nl_2d.nii.gz"
                    srv_base_rsl_crop_fn=f"{nl_2d_dir}/{brain}_{hemi}_{slab}_srv_rsl.nii.gz"
                        
                    init_align_rsl_fn=f"{nl_2d_dir}/{brain}_{hemi}_{slab}_init_align_{resolution}mm.nii.gz"

                    df_fn=f"reconstruction_output/1_init_align/{brain}_{hemi}_{slab}/final/{brain}_{hemi}_{slab}_final.csv"

                    nl_2d_vol=files[brain][hemi][slab][resolution]['nl_2d_vol']

                    direction=json.load(open(scale_factors_fn,'r'))[brain][hemi][slab]["direction"]

                    print('\tNL 2D volume:', nl_2d_vol, os.path.exists( nl_2d_vol ))
                    if not os.path.exists( nl_2d_vol ) :

                        if not os.path.exists(srv_base_rsl_crop_fn) or args.clobber :
                            shell(f'antsApplyTransforms -v 1 -d 3 -i {srv_base_rsl_fn} -r {init_align_fn} -t {nl_3d_tfm_inv_fn} -o {srv_base_rsl_crop_fn}', verbose=True)

                        receptor_2d_alignment( slab_df, init_align_fn, init_align_rsl_fn, srv_base_rsl_crop_fn, nl_2d_dir, nl_2d_vol, resolution, resolution_itr, direction=direction)
               
            ###
            ### Step 5 : Interpolate missing receptor densities using cortical surface mesh
            ###

            interp_dir=f'{out_dir}/5_surf_interp/'
            nl_2d_list = [ files[brain][hemi][i][resolution_list[-1]]['nl_2d_vol'] for i in slab_list ]
            nl_tfm_list = [ files[brain][hemi][i][resolution_list[-1]]['nl_3d_tfm_inv'] for i in slab_list ] 
            surface_interpolation(nl_tfm_list,  nl_2d_list, interp_dir, brain, hemi, resolution, df, n_depths=100)
