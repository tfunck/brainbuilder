import os
import json
import shutil
import re
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
from glob import glob
from scipy.ndimage import label
from scipy.ndimage import binary_dilation, binary_closing, binary_fill_holes
from scipy.ndimage.filters import gaussian_filter
from nibabel.processing import resample_to_output
from utils.utils import shell, create_2d_sections, run_stage, prefilter_and_downsample
from utils.ANTs import ANTs
from nibabel.processing import resample_to_output, smooth_image
from utils.mesh_io import load_mesh_geometry, save_mesh_data, save_obj, read_obj
from reconstruction.align_slab_to_mri import align_slab_to_mri
from reconstruction.receptor_segment import classifyReceptorSlices, resample_transform_segmented_images
from reconstruction.nonlinear_2d_alignment import receptor_2d_alignment, concatenate_sections_to_volume
from reconstruction.init_alignment import receptorRegister
from reconstruction.surface_interpolation import surface_interpolation
from reconstruction.crop import crop
from reconstruction.validate_reconstructed_sections import validate_reconstructed_sections
from preprocessing.preprocessing import fill_regions_3d

global file_dir
base_file_dir, fn =os.path.split( os.path.abspath(__file__) )
file_dir = base_file_dir +os.sep +'section_numbers' +os.sep

def create_new_srv_volumes(rec_3d_rsl_fn, srv_rsl_fn, cropped_output_list ):
    '''
    About : this function creates new gm srv images by removing gm from previously aligned slabs
            a single new volume is created at the highest resolution in the multiresolution
            heirarchy. the lower resolution srv images are created by downsampling the highest 
            resolution one
    Inputs : 
        rec_3d_rsl_fn :         aligned receptor volume in mni space at maximum 3d resolution
        srv_rsl_fn :            gm mask at current, maximum, 3d resolution
        cropped_output_list:    list of filenames for downsampled cropped srv fn files that will be used
                                for next slab
    Outputs :
        None 
    '''
    highest_res_srv_rsl_fn = cropped_output_list[-1]
    remove_slab_from_srv(rec_3d_rsl_fn, srv_rsl_fn, highest_res_srv_rsl_fn)
    print('Next srv fn:', highest_res_srv_rsl_fn)

    for i in range(len(cropped_output_list)-1) : #DEBUG resolution_list[0:-1] :
        #DEBUG
        #lower_res_srv_rsl_fn = next_slab_files[str(r)]['srv_crop_rsl_fn']
        lower_res_srv_rsl_fn = cropped_output_list[i]
        if not os.path.exists(lower_res_srv_rsl_fn) :
            r = resolution_list[i]
            prefilter_and_downsample(highest_res_srv_rsl_fn, float(r), lower_res_srv_rsl_fn, reference_image_fn=rec_3d_rsl_fn)

        print('Next srv fn:', lower_res_srv_rsl_fn)

def remove_slab_from_srv(slab_to_remove_fn, srv_rsl_fn, new_srv_rsl_fn):
    '''
    About :
        this function removes an aligned gm slab volume (slab_to_remove_fn) from the
        full gm volume (srv_rsl_fn)

    Inputs:
        slab_to_remove_fn : aligned volume to remove from gm 
        rec_3d_rsl_fn :     gm volume, potentially with previous slabs already removed
        new_srv_rsl_fn:     filename for new cropped srv rsl with aligned slab removed

    Outputs:
        None
    '''
    print(slab_to_remove_fn)
    print(srv_rsl_fn)
    #load slab gm mask, not strictly binary
    aligned_slab = nib.load(slab_to_remove_fn).get_fdata()
    
    #threshold aligned slab so that all non-zero values = 1
    aligned_slab[ aligned_slab > 0 ] = 1

    #load gm volume
    img = nib.load(srv_rsl_fn)
    vol = img.get_fdata()
    assert np.sum(aligned_slab) > 0 , f'Error: Empty volume in {slab_to_remove_fn}'
    
    #remove aligned slab from existing srv rsl volume.
    #needs to be thresholded because smoothing across the y-axis during downsampling
    #resutls in slab blurring too far across y-axis.
    
    #DEBUG
    print(vol.shape)
    print(aligned_slab.shape)
    vol[aligned_slab > 0. ] = 0 #probably conceptually simpler to threshold at 0, shouldnt make a practice


    #removing additional islands of tissue that may remain in srv volume
    vol_bin = np.zeros_like(vol)
    vol_bin[vol > 0.3 ] = 1

    assert np.sum(vol_bin) > 0 , f'Error: Empty binary volume for {slab_to_remove_fn}'
    vol_label, nlabels = label(vol_bin)
   
    #calculate the size of of the labeled regions in vol_label
    label_counts = np.bincount(vol_label.reshape(-1,))[1:]
   
    #find largest label
    max_label = np.max(label_counts)
    idx_to_keep = label_counts == max_label

    labels = np.arange(1,nlabels+1).astype(int)
   
    #create a mask volume that will only contain the largest region
    mask = np.zeros_like(vol)
    mask[vol_label==labels[idx_to_keep]]=1

    # use the fill_regions_3d function to fill in the GM
    # this basically means that the mask will fin in subcortical gm nuclei 
    # so that they don't get lost when multiplying by vol
    mask = fill_regions_3d(mask)

    vol *= mask

    print('\t\t\tWriting', new_srv_rsl_fn)
    nib.Nifti1Image(vol,img.affine).to_filename(new_srv_rsl_fn)

    

def resample_to_autoradiograph_sections(brain, hemi, slab, resolution, srv_rsl_fn, seg_rsl_fn, nl_3d_tfm_inv_fn, srv_space_rec_fn):
    '''
    About:
        Apply 3d transformation and resample volume into the same coordinate space as 3d receptor volume.           This produces a volume with 0.02mm dimension size along the y axis.

    Inputs:
        brain:      current subject brain id
        hemi:       current hemisphere (R or L)
        slab:       current slab number
        resolution:     current resolution level
        srv_rsl_fn:     gm super-resolution volume (srv) extracted from donor brain
        seg_rsl_fn:     brain mask of segmented autoradiographs
        nl_3d_tfm_inv_fn:   3d transformation from mni to receptor coordinate space
        srv_space_rec_fn:      
        
    Outpus:
        None
    '''
    temp_fn=f'/tmp/{brain}-{hemi}-{slab}.nii.gz'
    shell(f'antsApplyTransforms -v 1 -d 3 -i {srv_rsl_fn} -r {seg_rsl_fn} -t {nl_3d_tfm_inv_fn} -o {temp_fn}',True)
    img = nib.load(temp_fn)
    vol = img.get_fdata()

    img = resample_to_output(nib.Nifti1Image(vol,img.affine), [float(resolution),0.02, float(resolution)], order=5)
    img.to_filename(srv_space_rec_fn)
        

def calculate_section_order(autoradiograph_info_fn,  out_dir, in_df_fn='section_order/autoradiograph_info.csv') :
    '''
        About:
            create a csv file with information about each autoradiograph order in its slab
            volume
        Inputs:
            autoradiograph_info_fn: csv file with autoradiograph volume information
            out_dir :               output directory where csv file is saved
            in_df_fn :              pre-generated csv file with basic info about autoradiographs
        Outputs:
            None
    '''

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
    parser.add_argument('--mri-gm', dest='srv_fn', type=str, default='/data/receptor/human/mri1_R_gm_bg_srv.nii.gz', help='mri gm super-resolution volume (srv)')
    parser.add_argument('--surf-dir', dest='surf_dir', type=str, default='/data/receptor/human/MR1/civet/mri1/surfaces/', help='surface directory')
    parser.add_argument('--autoradiograph-info', dest='autoradiograph_info_fn', type=str, default=None, help='csv file with section info for each autoradiograph')
    parser.add_argument('--remote','-p', dest='remote', default=False, action='store_true',  help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--interpolation-only', dest='interpolation_only', default=False, action='store_true',  help='Slabs to reconstruct. Default = reconstruct all slabs.')

    return parser


def setup_files_json(args ):
    '''
    About:
        create a dictionary with all of the filenames that will be used in the reconstruction

    Inputs:
        args: user input arguments

    Outputs:
        files: return dictionary with all filenames
    '''


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
                    
                    if resolution == resolution_list[0]:
                        cdict['init_align_dir']=f'{args.out_dir}/{brain}_{hemi}_{slab}/1_init_align/'
                        cdict['init_align_fn']=cdict['init_align_dir'] + f'/brain-{brain}_hemi-{hemi}_slab-{slab}_init_align.nii.gz'

                    # Filenames
                    cdict['seg_rsl_fn']='{}/brain-{}_hemi-{}_slab-{}_seg_{}mm.nii.gz'.format(cdict['seg_dir'],brain, hemi, slab, resolution)
                    cdict['srv_rsl_fn'] = f'{args.out_dir}/{brain}_{hemi}_{slab}_mri_gm_{resolution}mm.nii.gz' 
                    cdict['srv_crop_rsl_fn'] = f'{args.out_dir}/{brain}_{hemi}_{slab}_mri_gm_crop_{resolution}mm.nii.gz' 

                    #if resolution_itr <= max_3d_itr  :
                    cdict['rec_3d_rsl_fn'] = '{}/{}_{}_{}_{}mm_rec_space-mri.nii.gz'.format(cdict['align_to_mri_dir'],brain,hemi,slab,resolution)
                    cdict['srv_3d_rsl_fn'] = '{}/{}_{}_{}_{}mm_mri_gm_space-rec.nii.gz'.format(cdict['align_to_mri_dir'],brain,hemi,slab,resolution)
                    cdict['nl_3d_tfm_fn'] = '{}/rec_to_mri_SyN_Composite.h5'.format(cdict['align_to_mri_dir'])
                    cdict['nl_3d_tfm_inv_fn'] = '{}/rec_to_mri_SyN_InverseComposite.h5'.format(cdict['align_to_mri_dir'])

                    cdict['nl_2d_vol_fn'] = "{}/{}_{}_{}_nl_2d_{}mm.nii.gz".format(cdict['nl_2d_dir'] ,brain,hemi,slab,resolution) 
                    cdict['srv_space_rec_fn'] = "{}/{}_{}_{}_srv_rsl.nii.gz".format(cdict['nl_2d_dir'], brain, hemi, slab)   
                    #if resolution_itr == max_3d_itr :  
                    #    max_3d_cdict=cdict
                    files[brain][hemi][slab][resolution]=cdict
    json.dump(files,open(args.files_json,'w+'))
    return files

def setup_parameters(args) : 
    '''
        About:
            Setup the parameters and filenames that will be used in the reconstruction
        Inputs:
            args:   user input arguments

        Outputs:
            files: return dictionary with all filenames
            args:   user input arguments with some additional parameters tacked on in this function
    '''
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
    '''
    About:
        Mutliresolution scheme that a. segments autoradiographs, b. aligns these to the donor mri in 3D,
        and c. aligns autoradiograph sections to donor MRI in 2d. This schema is repeated for each 
        resolution in the resolution heiarchy.  

    Inputs:
        slab_df:    data frame with info for autoradiographs in current slab
        hemi_df     data frame with info for autoradiographs in current hemisphere
        brain:       name of current subject id
        hemi:        current hemisphere
        slab:        current slab
        args:        user input arguments
        files:       dictionary with all the filenames used in reconstruction
        resolution_list:    heirarchy of resolutions  
        init_align_fn:      filename for initial alignement of autoradiographs
        max_resolution:     maximum 3d resolution that can be used

    Outputs:
        slab_df: updated slab_df data frame with filenames for nonlinearly 2d aligned autoradiographs

    '''

    slab_list = [int(i) for i in  files[brain][hemi].keys() ]
    
    ### Iterate over progressively finer resolution
    for resolution_itr, resolution in enumerate(resolution_list) :
        print('\tMulti-Resolution Alignement:',resolution)
        cfiles = files[brain][hemi][str(slab)][str(resolution)] #Current files
        
        cur_out_dir = cfiles['cur_out_dir']
        seg_dir = cfiles['seg_dir'] 
        align_to_mri_dir = cfiles['align_to_mri_dir'] 
        nl_2d_dir = cfiles['nl_2d_dir']

        srv_rsl_fn = cfiles['srv_rsl_fn']  
        srv_crop_rsl_fn = cfiles['srv_crop_rsl_fn']  

        seg_rsl_fn = cfiles['seg_rsl_fn']
        nl_3d_tfm_fn = cfiles['nl_3d_tfm_fn']
        nl_3d_tfm_inv_fn = cfiles['nl_3d_tfm_inv_fn']
        rec_3d_rsl_fn = cfiles['rec_3d_rsl_fn']
        srv_3d_rsl_fn = cfiles['srv_3d_rsl_fn']
        srv_space_rec_fn = cfiles['srv_space_rec_fn']
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
        ### Stage 1.25 : Downsample SRV to current resolution
        ###
        print('\t\tStage 1.25' )
        crop_srv_rsl_fn = files[brain][hemi][str(int(slab))][str(resolution)]['srv_crop_rsl_fn']
        if run_stage([args.srv_fn], [srv_rsl_fn, crop_srv_rsl_fn]) or args.clobber :
            # downsample the original srv gm mask to current 3d resolution
            prefilter_and_downsample(args.srv_fn, resolution_3d, srv_rsl_fn)
            
            # if this is the fiest slab, then the cropped version of the gm mask
            # is the same as the downsampled version because there are no prior slabs to remove.
            # for subsequent slabs, the cropped srv file is created at stage 3.5
            if int(slab) == int(slab_list[0]) : shutil.copy(srv_rsl_fn, crop_srv_rsl_fn)

        ### Stage 1.5 : combine aligned sections from previous iterations into a volume
        #Combine 2d sections from previous resolution level into a single volume
        if resolution != resolution_list[0] and not os.path.exists(last_nl_2d_vol_fn)  :
            last_nl_2d_dir = files[brain][hemi][slab][prev_resolution]['nl_2d_dir']
            concatenate_sections_to_volume(slab_df, init_align_fn, last_nl_2d_dir, last_nl_2d_vol_fn)

        ###
        ### Stage 2 : Autoradiograph segmentation
        ###
        print('\t\tStep 2: Autoradiograph segmentation')
        stage_2_outputs=[seg_rsl_fn]
        if not os.path.exists(seg_rsl_fn) or args.clobber  :
            resample_transform_segmented_images(slab_df, resolution, resolution_3d, seg_dir+'/2d/' )
            #write 2d segmented sections at current resolution. apply initial transform
            classifyReceptorSlices(slab_df, align_fn, seg_dir+'/2d/', seg_dir, seg_rsl_fn, resolution=resolution_3d )

        ###
        ### Stage 3 : Align slabs to MRI
        ###
        print('\t\tStep 3: align slabs to MRI')
        stage_3_outputs = [nl_3d_tfm_fn, nl_3d_tfm_inv_fn, rec_3d_rsl_fn, srv_3d_rsl_fn]

        if run_stage(stage_2_outputs, stage_3_outputs) or args.clobber  :
            print('Cropped rsl fn', crop_srv_rsl_fn)
            align_slab_to_mri(seg_rsl_fn, crop_srv_rsl_fn, slab, align_to_mri_dir, hemi_df, args.slabs, nl_3d_tfm_fn, nl_3d_tfm_inv_fn, rec_3d_rsl_fn, srv_3d_rsl_fn, resolution_3d, resolution_itr_3d, resolution_list  )
        
        ###
        ### Stage 3.5 : Create a new srv_rsl_fn file that removes the currently aligned slab
        ###
        #check that the current slab isn't the last one
        not_last_slab = int(slab) < max(slab_list)
        if not_last_slab  and resolution == resolution_list[-1] :
            # filenames for the next slab
            next_slab_files = files[brain][hemi][str(int(slab)+1)]
            # filename for cropped srv file for next slab
            #highest_res_srv_rsl_fn = next_slab_files[str(resolution)]['srv_crop_rsl_fn']
            stage_3_5_outputs = [ next_slab_files[str(r)]['srv_crop_rsl_fn']   for r in resolution_list ]
            #if run_stage( stage_3_outputs, [highest_res_srv_rsl_fn]) :
            if run_stage( stage_3_outputs, stage_3_5_outputs):
                print('\t\tStage 3.5 : Removing aligned slab from srv')
                create_new_srv_volumes(rec_3d_rsl_fn, crop_srv_rsl_fn, stage_3_5_outputs)

        
        ###
        ### Stage 4 : 2D alignment of receptor to resample MRI GM vol
        ###
        if run_stage([srv_rsl_fn], [srv_space_rec_fn]) or args.clobber :
            resample_to_autoradiograph_sections(brain, hemi, slab, float(resolution), srv_rsl_fn, seg_rsl_fn, nl_3d_tfm_inv_fn, srv_space_rec_fn)
        
        create_2d_sections( slab_df, srv_space_rec_fn, float(resolution), nl_2d_dir )
            
        print('\t\tStep 4: 2d nl alignment')
        stage_4_outputs=[nl_2d_vol_fn]
        if run_stage(stage_3_outputs, stage_4_outputs)  or args.clobber:
            slab_df = receptor_2d_alignment( slab_df, init_align_fn, srv_space_rec_fn,seg_dir+'/2d/', nl_2d_dir,  resolution, resolution_itr)
            #Concatenate 2D nonlinear aligned sections into output volume
            concatenate_sections_to_volume(slab_df, srv_space_rec_fn, nl_2d_dir, nl_2d_vol_fn)

    #slab_df.to_csv(f'{cur_out_dir}/file_info_{resolution}.csv')
    return slab_df

            

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
    for slab in args.slabs :
        print('Slab:', slab)
        slab_df=df.loc[(df['hemisphere']==hemi) & (df['mri']==brain) & (df['slab']==int(slab)) ]
        init_align_fn=files[brain][hemi][str(slab)][str(resolution_list[0])]['init_align_fn']
        init_align_dir=files[brain][hemi][str(slab)][resolution_list[0]]['init_align_dir']
        init_tfm_csv =f'{init_align_dir}/{brain}_{hemi}_{slab}_final.csv'
        slab_tfm_csv =f'{init_align_dir}/{brain}_{hemi}_{slab}_slab_tfm.csv'

        ###  Step 1: Initial Alignment
        print('\tInitial rigid inter-autoradiograph alignment')
        if (not os.path.exists( init_align_fn) or not os.path.exists(init_tfm_csv) or args.clobber) :
            receptorRegister(brain, hemi, slab, init_align_fn, init_tfm_csv, init_align_dir, slab_df, scale_factors_json=args.scale_factors_fn, clobber=args.clobber)
        
        if not os.path.exists(slab_tfm_csv): 
            slab_df = add_tfm_column(slab_df, init_tfm_csv,slab_tfm_csv)
        else : slab_df = pd.read_csv(slab_tfm_csv)

        ### Steps 2-4 : Multiresolution alignment
        slab_df = multiresolution_alignment(slab_df, hemi_df, brain, hemi, slab, args,files, resolution_list, init_align_fn)
        '''
        def create_brain_mask_sections(slab_df, slab, resolution, tfm_3d_inv_fn, reference_fn, brain_mask_fn):
            brain_mask_rsl_fn = re.sub('.nii.gz','_space-rec-{slab}.nii.gz')  
            shell(f'antsApplyTransforms -i {brain_mask_fn} -t {tfm_3d_inv_fn} -r {reference_fn} -o {brain_mask_rsl_fn}')
            brain_mask_dir = 
            create_2d_sections( slab_df, brain_mask_rsl_fn, float(resolution), brain_mask_dir )

            for row in slab_df.iterrows():
            
                shell('antsApplyTransforms -i {brain_section_fn}'
        '''

    ###
    ### Step 5 : Interpolate missing receptor densities using cortical surface mesh
    ###
    interp_dir=f'{args.out_dir}/5_surf_interp/'

    slab_dict={} 
    for slab, temp_slab_dict in files[brain][hemi].items() :
        nl_3d_tfm_exists = os.path.exists(temp_slab_dict[resolution_list[-1]]['nl_3d_tfm_fn'])
        nl_2d_vol_exists = os.path.exists(temp_slab_dict[resolution_list[-1]]['nl_2d_vol_fn'])
        if nl_3d_tfm_exists and nl_2d_vol_exists :
            slab_dict[slab] = temp_slab_dict[resolution_list[-1]] 
        else : 
            print(f'Error: not including {slab} for interpolation (nl 3d tfm exists = {nl_3d_tfm_exists}, 2d nl vol exists = {nl_2d_vol_exists}) ')
            exit(0)
    
    ###
    ### 6. Surface interpolation
    ###
    assert len(slab_dict.keys()) != 0 , print('No slabs to interpolate over')
    ligand='flum'
    ligand_df = hemi_df.loc[ ligand  == hemi_df['ligand'] ]
    surface_interpolation(slab_dict, args.out_dir, interp_dir, brain, hemi, highest_resolution, ligand_df, args.srv_fn, surf_dir=args.surf_dir, n_vertices=args.n_vertices, n_depths=args.n_depths)

    ###
    ### 7. Quality Control
    ###
    max_resolution = resolution_list[-1]
    depth = '0.45'
    validate_reconstructed_sections(max_resolution, args.n_depths, ligand_df, base_out_dir='/data/receptor/human/output_2/', clobber=False)
    #FIXME filename should be passed from surface_interpolation
    ligand_csv = glob(f'{interp_dir}/*{ligand}*{depth}*_raw.csv')[0]   
    sphere_mesh_fn = glob(f'{interp_dir}/surfaces/surf_{max_resolution}mm_{depth}_inflate_rsl.h5')[0]
    cortex_mesh_fn = glob(f'{interp_dir}/surfaces/surf_{max_resolution}mm_{depth}_rsl.h5')[0]
    #validate_interpolation(ligand_csv, sphere_mesh_fn, cortex_mesh_fn, n_samples=10000, max_depth=5)


###---------------------###
###  PROCESSING STEPS   ###
###---------------------###
#   0. Crop all autoradiographs
#   1. Init Alignment (Rigid 2D, per slab)
#   2. GM segmentation of receptor volumes (per slab)
#   3. GM MRI to autoradiograph volume (Nonlinear 3D, per slab)
#   4. Autoradiograph to GM MRI (2D nonlinear, per slab)
#   4.5. Remove aligned slab from SRV GM volume
#   5. Interpolate missing vertices on sphere, interpolate back to 3D volume

if __name__ == '__main__':
    resolution_list = ['4.0', '3.5', '3.0', '2.5', '2.0', '1.5', '1.0', '0.8', '0.6', '0.4'] #, '0.2'] #, '0.05' ]

    args, files = setup_parameters(setup_argparse().parse_args() )
    #Process the base autoradiograph csv
    if not os.path.exists(args.autoradiograph_info_fn) : 
        calculate_section_order(args.autoradiograph_info_fn, args.crop_dir, args.out_dir, in_df_fn=file_dir+os.sep+'autoradiograph_info.csv')

    df = pd.read_csv(args.autoradiograph_info_fn)
    
    ### Step 0 : Crop downsampled autoradiographs
    crop(args.src_dir, args.mask_dir, args.out_dir, df, args.scale_factors_fn, remote=args.remote)
    
    for brain in args.brain :
        for hemi in args.hemi :                     
            print('Brain:',brain,'Hemisphere:', hemi)
            reconstruct_hemisphere(df, brain, hemi,  args, files, resolution_list)

