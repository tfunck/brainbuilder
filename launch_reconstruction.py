import os
import json
import re
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
from scipy.ndimage.filters import gaussian_filter
from nibabel.processing import resample_to_output
from utils.utils import shell, resample, create_2d_sections
from utils.ANTs import ANTs
from nibabel.processing import resample_to_output, smooth_image
from utils.mesh_io import load_mesh_geometry, save_mesh_data, save_obj, read_obj
from reconstruction.align_slab_to_mri import align_slab_to_mri
from reconstruction.receptor_segment import classifyReceptorSlices, resample_transform_segmented_images
from reconstruction.nonlinear_2d_alignment import receptor_2d_alignment, concatenate_sections_to_volume
from reconstruction.init_alignment import receptorRegister
from reconstruction.surface_interpolation import surface_interpolation, surf_fn_str
from reconstruction.crop import crop


global file_dir
base_file_dir, fn =os.path.split( os.path.abspath(__file__) )
file_dir = base_file_dir +os.sep +'section_numbers' +os.sep

def w2v(c, step, start):
    return np.round( (c-start)/step ).astype(int)

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

    df['crop_fn'] = df['lin_fn'].apply(lambda x: out_dir+'/0_crop/'+os.path.splitext(os.path.basename(x))[0]+'#L.nii.gz')
    df['seg_fn'] = df['lin_fn'].apply(lambda x: out_dir+'/0_crop/'+os.path.splitext(os.path.basename(x))[0]+'_seg.nii.gz')

    df.sort_values(["mri","hemisphere","slab","volume_order"], inplace=True)

    # Remove UB , "unspecific binding" from the df
    df = df.loc[df['repeat'] != 'UB']

    df['lin_base_fn'] = df['lin_fn'].apply( lambda x: os.path.splitext(os.path.basename(x))[0] ) 

    df.to_csv(args.autoradiograph_info_fn)

def setup_argparse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--brains','-b', dest='brain', nargs='+', default=['MR1'], help='Brains to reconstruct. Default = run all.')
    parser.add_argument('--hemispheres', '--hemi', dest='hemi',default=['R'], nargs='+', help='Brains to reconstruct. Default = reconstruct all hemispheres.')
    parser.add_argument('--ligand', dest='ligand',default='afdx', help='Ligand to reconstruct for final volume.')
    parser.add_argument('--clobber', dest='clobber',default=False,action='store_true', help='Overwrite existing results')
    parser.add_argument('--slab','-s', dest='slab', nargs='+', default=[1], help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--chunk-perc','-u', dest='slab_chunk_perc', type=float, default=1., help='Subslab size (use with --nonlinear-only option) ')
    parser.add_argument('--chunk','-c', dest='slab_chunk_i', type=int, default=1, help='Subslab to align (use with --nonlinear-only option).')
    parser.add_argument('--nvertices', dest='n_vertices', type=int, default=81920, help='n vertices for mesh')
    parser.add_argument('--ndepths', dest='n_depths', type=int, default=10, help='n depths for mesh')
    parser.add_argument('--src-dir','-i', dest='src_dir', type=str, default='receptor_dwn', help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--mask-dir', dest='mask_dir', type=str, default='/data/receptor/human/crop/', help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--out-dir','-o', dest='out_dir', type=str, default='output', help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--scale-factors', dest='scale_factors_fn', type=str, default=None, help='json file with scaling and ordering info for each slab')
    parser.add_argument('--mri-gm', dest='srv_fn', type=str, default='/data/receptor/human/mri1_gm_bg_srv.nii.gz', help='mri gm super-resolution volume (srv)')
    parser.add_argument('--surf-dir', dest='surf_dir', type=str, default='/data/receptor/human/MR1/civet/mri1/surfaces/', help='surface directory')
    parser.add_argument('--autoradiograph-info', dest='autoradiograph_info_fn', type=str, default=None, help='csv file with section info for each autoradiograph')
    parser.add_argument('--remote','-p', dest='remote', default=False, action='store_true',  help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--interpolation-only', dest='interpolation_only', default=False, action='store_true',  help='Slabs to reconstruct. Default = reconstruct all slabs.')

    return parser


def setup_files_json(args, max_3d_resolution=0.3):
    print(args.files_json)
    #max_3d_itr=len(resolution_list)
    #for resolution_itr, resolution in enumerate(resolution_list) :
    #    if float(resolution) < max_3d_resolution :
    #        max_3d_itr = resolution_itr-1
    #        assert max_3d_itr >= 0 , 'max_3d_resolution is set too high '
    #        break


    files={}
    for brain in args.brain :
        files[brain]={}
        for hemi in args.hemi :
            files[brain][hemi]={}
            for slab in args.slabs : 
                files[brain][hemi][slab]={}

                for resolution_itr, resolution in enumerate(resolution_list) :
                    cdict={} #current dictionary
                    cdict['brain']=brain        
                    cdict['hemi']=hemi        
                    cdict['slab']=slab        

                    # Directories
                    cdict['cur_out_dir']=f'{args.out_dir}/{brain}_{hemi}_{slab}/{resolution}mm/'
                    cdict['seg_dir']='{}/2_segment/'.format(cdict['cur_out_dir'])
                    cdict['align_to_mri_dir']='{}/3_align_slab_to_mri/'.format(cdict['cur_out_dir'])
                    cdict['nl_2d_dir']='{}/4_nonlinear_2d'.format(cdict['cur_out_dir'])
                    print(resolution, resolution_list[0] ) 
                    if resolution == resolution_list[0]:
                        cdict['init_align_dir']=f'{args.out_dir}/{brain}_{hemi}_{slab}/1_init_align/'
                        cdict['init_align_fn']=cdict['init_align_dir'] + f'/brain-{brain}_hemi-{hemi}_slab-{slab}_init_align.nii.gz'

                    # Filenames
                    cdict['seg_rsl_fn']='{}/brain-{}_hemi-{}_slab-{}_seg_{}mm.nii.gz'.format(cdict['seg_dir'],brain, hemi, slab, resolution)
                    cdict['srv_rsl_fn'] = f'{args.out_dir}/{brain}_{hemi}_mri_gm_{resolution}mm.nii.gz' 

                    #if resolution_itr <= max_3d_itr  :
                    cdict['rec_3d_rsl_fn'] = '{}/{}_{}_{}_rec_space-mri.nii.gz'.format(cdict['align_to_mri_dir'],brain,hemi,slab)
                    cdict['srv_3d_rsl_fn'] = '{}/{}_{}_{}_mri_gm_space-rec.nii.gz'.format(cdict['align_to_mri_dir'],brain,hemi,slab)
                    cdict['nl_3d_tfm_fn'] = '{}/rec_to_mri_SyN_Composite.h5'.format(cdict['align_to_mri_dir'])
                    cdict['nl_3d_tfm_inv_fn'] = '{}/rec_to_mri_SyN_InverseComposite.h5'.format(cdict['align_to_mri_dir'])

                    cdict['nl_2d_vol_fn'] = "{}/{}_{}_{}_nl_2d_{}mm.nii.gz".format(cdict['nl_2d_dir'] ,brain,hemi,slab,resolution) 
                    cdict['srv_base_rsl_crop_fn'] = "{}/{}_{}_{}_srv_rsl.nii.gz".format(cdict['nl_2d_dir'], brain, hemi, slab)   
                    #if resolution_itr == max_3d_itr :  
                    #    max_3d_cdict=cdict
                    files[brain][hemi][slab][resolution]=cdict
    json.dump(files,open(args.files_json,'w+'))
    return files

def setup_parameters(args) : 
    ###
    ### Parameters
    ###
    args.slabs=['1','2','3','4','5','6'] #FIXME shouldnt be hard coded

    if args.scale_factors_fn == None :
        args.scale_factors_fn=base_file_dir+'/scale_factors.json'

    if args.autoradiograph_info_fn == None :
        args.autoradiograph_info_fn=args.out_dir+'/autoradiograph_info_volume_order.csv'

    if args.srv_fn == None :
        args.srv_fn="srv/mri1_gm_bg_srv.nii.gz"

    args.crop_dir=f'{args.out_dir}/0_crop'
    os.makedirs(args.crop_dir,exist_ok=True)

    args.files_json=args.out_dir+"/reconstruction_files.json"
    files = setup_files_json(args)


    return args, files 



def multiresolution_alignment(slab_df,  hemi_df, brain, hemi, slab, args, files, resolution_list, init_align_fn, max_resolution=0.3):
    ### Iterate over progressively finer resolution
    for resolution_itr, resolution in enumerate(resolution_list) :
        print('Resolution',resolution)
        cfiles = files[brain][hemi][str(slab)][str(resolution)] #Current files
        
        cur_out_dir = cfiles['cur_out_dir']
        seg_dir = cfiles['seg_dir'] 
        align_to_mri_dir = cfiles['align_to_mri_dir'] 
        nl_2d_dir = cfiles['nl_2d_dir']

        srv_rsl_fn = cfiles['srv_rsl_fn']  
        seg_rsl_fn = cfiles['seg_rsl_fn']
        nl_3d_tfm_fn = cfiles['nl_3d_tfm_fn']
        nl_3d_tfm_inv_fn = cfiles['nl_3d_tfm_inv_fn']
        rec_3d_rsl_fn = cfiles['rec_3d_rsl_fn']
        srv_3d_rsl_fn = cfiles['srv_3d_rsl_fn']
        srv_base_rsl_crop_fn = cfiles['srv_base_rsl_crop_fn']
        nl_2d_vol_fn = cfiles['nl_2d_vol_fn']

        resolution_3d = max(float(resolution), max_resolution)
        if resolution_3d == max_resolution :
            resolution_itr_3d = [ i for i, res in enumerate(resolution_list) if float(res) >= max_resolution ][-1]
        else :
            resolution_itr_3d = resolution_itr

        for dir_name in [cur_out_dir, align_to_mri_dir , seg_dir, nl_2d_dir ] :
            os.makedirs(dir_name, exist_ok=True)

        prev_resolution=resolution_list[resolution_itr-1]

        last_nl_2d_vol_fn = files[brain][hemi][str(slab)][str(resolution_list[resolution_itr-1])]['nl_2d_vol_fn']
        if resolution_itr > 0 : 
            align_fn = last_nl_2d_vol_fn
        else : 
            align_fn = init_align_fn

        ###
        ### Step 1.5 : Downsample SRV to current resolution
        ###
        if not os.path.exists(srv_rsl_fn) or args.clobber :
            resample_to_output(nib.load(args.srv_fn), [float(resolution_3d)]*3).to_filename( srv_rsl_fn)
        
        #Combine 2d sections from previous resolution level into a single volume
        if (resolution != resolution_list[0] and not os.path.exists(last_nl_2d_vol_fn))  :
            last_nl_2d_dir = files[brain][hemi][slab][prev_resolution]['nl_2d_dir']
            concatenate_sections_to_volume(slab_df, init_align_fn, last_nl_2d_dir, last_nl_2d_vol_fn)

        ###
        ### Step 2 : Autoradiograph segmentation
        ###
        print('\tStep 2: Autoradiograph segmentation')
        if (not os.path.exists(seg_rsl_fn) or args.clobber)  :
            resample_transform_segmented_images(slab_df, resolution, resolution_3d, seg_dir+'/2d/' )
            #write 2d segmented sections at current resolution. apply initial transform
            classifyReceptorSlices(slab_df, align_fn, seg_dir+'/2d/', seg_dir, seg_rsl_fn, resolution=resolution_3d )

        ###
        ### Step 3 : Align slabs to MRI
        ###
        print('\tStep 3: align slabs to MRI')
        dir_list = [nl_3d_tfm_fn, nl_3d_tfm_inv_fn, rec_3d_rsl_fn, srv_3d_rsl_fn]
        if False in [ os.path.exists(fn) for fn in dir_list ]  :
            align_slab_to_mri(seg_rsl_fn, srv_rsl_fn, slab, align_to_mri_dir, hemi_df, args.slabs, nl_3d_tfm_fn, nl_3d_tfm_inv_fn, rec_3d_rsl_fn, srv_3d_rsl_fn, resolution_3d, resolution_itr_3d, resolution_list  )

        ###
        ### Step 4 : 2D alignment of receptor to resample MRI GM vol
        ###
        if not os.path.exists(srv_base_rsl_crop_fn) or args.clobber :
            temp_fn=f'/tmp/{brain}-{hemi}-{slab}.nii.gz'
            shell(f'antsApplyTransforms -v 0 -d 3 -i {srv_rsl_fn} -r {seg_rsl_fn} -t {nl_3d_tfm_inv_fn} -o {temp_fn}')
            img = nib.load(temp_fn)
            vol = img.get_fdata()
            vol = gaussian_filter(vol, float(resolution) / (0.2 * 2))

            img = resample_to_output(nib.Nifti1Image(vol,img.affine), [float(resolution),0.02, float(resolution)], order=5)
            img.to_filename(srv_base_rsl_crop_fn)
        
        create_2d_sections( slab_df, init_align_fn, srv_base_rsl_crop_fn, float(resolution), nl_2d_dir )
        #Check if necessary files exist to proceed to 2d nl alignment
        exit_early=False
        for fn in [ init_align_fn, srv_base_rsl_crop_fn, nl_2d_dir ] : 
            if not os.path.exists(fn) :
                print('Error: could not run 2d nonlinear alignment, missing', fn)
                exit_early=True
        if exit_early : exit(1)
            
        print('\tStep 4: 2d nl alignment')
        slab_df = receptor_2d_alignment( slab_df, init_align_fn, srv_base_rsl_crop_fn,seg_dir+'/2d/', nl_2d_dir,  resolution, resolution_itr)
   
        #Concatenate 2D nonlinear aligned sections into output volume
        if  not os.path.exists(nl_2d_vol_fn)  :
            concatenate_sections_to_volume(slab_df, srv_base_rsl_crop_fn, nl_2d_dir, nl_2d_vol_fn)



def add_tfm_column(slab_df, init_tfm_csv, slab_tfm_csv) :
    tfm_df = pd.read_csv(init_tfm_csv)
    slab_df['init_tfm'] = [None] * slab_df.shape[0]

    for i, row in tfm_df.iterrows():
        slab_df['init_tfm'].loc[ slab_df['global_order'] == row['global_order'] ] = row['init_tfm']
   
    slab_df['tfm'] = slab_df['init_tfm']

    slab_df.to_csv(slab_tfm_csv)
    return slab_df

def reconstruct_hemisphere(df, brain, hemi, args, files, resolution_list):
    hemi_df = df.loc[ (df['mri']==brain) & (df['hemisphere']==hemi) ]
    highest_resolution=resolution_list[-1]

    ### Reconstruct slab
    for slab in args.slab :
        slab_df=df.loc[(df['hemisphere']==hemi) & (df['mri']==brain) & (df['slab']==int(slab)) ]
        init_align_fn=files[brain][hemi][str(slab)][str(resolution_list[0])]['init_align_fn']
        init_align_dir=files[brain][hemi][str(slab)][resolution_list[0]]['init_align_dir']
        init_tfm_csv =f'{init_align_dir}/{brain}_{hemi}_{slab}_final.csv'
        slab_tfm_csv =f'{init_align_dir}/{brain}_{hemi}_{slab}_slab_tfm.csv'

        if not os.path.exists(slab_tfm_csv): 
            slab_df = add_tfm_column(slab_df, init_tfm_csv,slab_tfm_csv)
        else : slab_df = pd.read_csv(slab_tfm_csv)

        ###  Step 1: Initial Alignment
        print('\tInitial rigid inter-autoradiograph alignment')
        if (not os.path.exists( init_align_fn) or not os.path.exists(init_tfm_csv) or args.clobber) :
            receptorRegister(brain, hemi, slab, init_align_fn, init_tfm_csv, init_align_dir, slab_df, scale_factors_json=args.scale_factors_fn, clobber=args.clobber)

        ### Steps 2-4 : Multiresolution alignment
        multiresolution_alignment(slab_df, hemi_df, brain, hemi, slab, args,files, resolution_list,  init_align_fn)

    ### Step 5 : Interpolate missing receptor densities using cortical surface mesh
    interp_dir=f'{args.out_dir}/5_surf_interp/'

    slab_dict={} 
    for slab, temp_slab_dict in files[brain][hemi].items() :
        if os.path.exists(temp_slab_dict[resolution_list[-1]]['nl_3d_tfm_fn']) and os.path.exists(temp_slab_dict[resolution_list[-1]]['nl_2d_vol_fn']) :
            slab_dict[slab] = temp_slab_dict[resolution_list[-1]] 

    if len(slab_dict.keys()) == 0 : 
        print('No slabs to interpolate over')
        exit(0)
    
    # Surface interpolation
    ligand_df = hemi_df.loc[ args.ligand == hemi_df['ligand'] ]
    surface_interpolation(slab_dict, args.out_dir, interp_dir, brain, hemi, highest_resolution, ligand_df, args.srv_fn, surf_dir=args.surf_dir, n_vertices=args.n_vertices, n_depths=args.n_depths)


###---------------------###
###  PROCESSING STEPS   ###
###---------------------###
#   0. Crop all autoradiographs
#   1. Init Alignment (Rigid 2D, per slab)
#   2. GM segmentation of receptor volumes (per slab)
#   3. GM MRI to autoradiograph volume (Nonlinear 3D, per slab)
#   4. Autoradiograph to GM MRI (2D nonlinear, per slab)
#   5. Interpolate missing vertices on sphere, interpolate back to 3D volume

if __name__ == '__main__':
    resolution_list = [ '2.4', '1.6'] #, '0.8'] #, '0.4']#, '0.2'] #, '0.05' ]

    args, files = setup_parameters(setup_argparse().parse_args() )
    #Process the base autoradiograph csv
    if not os.path.exists(args.autoradiograph_info_fn) : 
        calculate_section_order(args.autoradiograph_info_fn, args.crop_dir, args.out_dir, in_df_fn=file_dir+os.sep+'autoradiograph_info.csv')

    df = pd.read_csv(args.autoradiograph_info_fn)
    
    ### Step 0 : Crop downsampled autoradiographs
    crop(args.src_dir, args.mask_dir, args.out_dir, df, args.scale_factors_fn, remote=args.remote)

    for brain in args.brain :
        for hemi in args.hemi :                     
            reconstruct_hemisphere(df, brain, hemi,  args, files, resolution_list)

