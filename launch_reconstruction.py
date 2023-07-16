import os
import json
import shutil
import re
import argparse
import pandas as pd
import numpy as np
import nibabel
import utils.ants_nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import debug; 
from copy import deepcopy
from skimage.transform import resize
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from validation.validate_alignment import validate_alignment
from reconstruction.multiresolution_alignment import multiresolution_alignment
from glob import glob
from scipy.ndimage import label
from scipy.ndimage import binary_dilation, binary_closing, binary_fill_holes
from utils.utils import shell,  run_stage, prefilter_and_downsample, resample_to_autoradiograph_sections, num_cores
from utils.ANTs import ANTs
from utils.mesh_io import load_mesh_geometry, save_mesh_data, save_obj, read_obj
from utils.reconstruction_classes import SlabReconstructionData
from reconstruction.init_alignment import receptorRegister, apply_transforms_to_landmarks
from reconstruction.surface_interpolation import  surface_interpolation, create_thickened_volumes
from reconstruction.crop import crop, process_landmark_images
from validation.validate_interpolation import validate_interpolation, plot_r2
from validation.validate_reconstructed_sections import validate_reconstructed_sections
from preprocessing.preprocessing import fill_regions_3d
#from reconstruction.batch_correction import correct_batch_effects 

global file_dir
base_file_dir, fn =os.path.split( os.path.abspath(__file__) )
file_dir = base_file_dir +os.sep +'section_numbers' +os.sep
manual_dir = base_file_dir +os.sep +'manual_points' +os.sep

if debug.DEBUG > 0: print('\n\nRUNNING WITH DEBUG',debug.DEBUG,'\n\n')

def create_new_srv_volumes(rec_3d_rsl_fn, srv_rsl_fn, cropped_output_list, resolution_list ):
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
    highest_res_srv_rsl_fn = srv_rsl_fn #DEBUG #cropped_output_list[-1]
    #DEBUG remove_slab_from_srv(rec_3d_rsl_fn, srv_rsl_fn, highest_res_srv_rsl_fn)
    
    print('Next srv fn:', highest_res_srv_rsl_fn)

    #for i in range(len(cropped_output_list)-1) : #DEBUG resolution_list[0:-1] :
    for i in range(len(cropped_output_list)) : #DEBUG 
        lower_res_srv_rsl_fn = cropped_output_list[i]
        print('\tCreating', lower_res_srv_rsl_fn)
        if not os.path.exists(lower_res_srv_rsl_fn) :
            r = resolution_list[i]
            prefilter_and_downsample(highest_res_srv_rsl_fn, [float(r)]*3, lower_res_srv_rsl_fn) #, reference_image_fn=rec_3d_rsl_fn)

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
    print('slab_to_remove_fn', slab_to_remove_fn)
    print('SRV', srv_rsl_fn)
    print('New SRV RSL', new_srv_rsl_fn)
    shutil.copy(srv_rsl_fn, new_srv_rsl_fn)
    return None
    
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
    #resutls in slab blurring too far across y-axis

    aligned_slab = resize(aligned_slab, vol.shape, order=0)

    vol[aligned_slab > 0. ] = 0 
    
    #nib.Nifti1Image(vol,img.affine,direction_order='lpi').to_filename(new_srv_rsl_fn) #DEBUG
    #return None #DEBUG

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
    #vol = np.flip(vol, axis=1)
    nib.Nifti1Image(vol,img.affine,direction_order='lpi').to_filename(new_srv_rsl_fn)

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
    df['crop_raw_fn'] = df['crop_fn']
    df['seg_fn'] = df['lin_fn'].apply(lambda x: out_dir+'/0_crop/'+os.path.splitext(os.path.basename(x))[0]+'_seg.nii.gz')
    df['pseudo_cls_fn'] = df['lin_fn'].apply(lambda x: out_dir+'/0_crop/'+os.path.splitext(os.path.basename(x))[0]+'_pseudo-cls.nii.gz')

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
    parser.add_argument('--debug', dest='debug',default=False,action='store_true', help='DEBUG mode')

    parser.add_argument('--slab','-s', dest='slab', nargs='+', default=[1], help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--resolutions','-r', dest='resolution_list', nargs='+', default=[4.0, 3.0, 2.0, 1.0, 0.5, 0.25],type=float, help='Resolution list.')
    parser.add_argument('--chunk-perc','-u', dest='slab_chunk_perc', type=float, default=1., help='Subslab size (use with --nonlinear-only option) ')
    parser.add_argument('--chunk','-c', dest='slab_chunk_i', type=int, default=1, help='Subslab to align (use with --nonlinear-only option).')
    parser.add_argument('--nvertices', dest='n_vertices', type=int, default=81920, help='n vertices for mesh')
    parser.add_argument('--ndepths', dest='n_depths', type=int, default=10, help='n depths for mesh')
    parser.add_argument('--no-surf', dest='no_surf', action='store_true', default=False, help='Exit after multi-resolution alignment')
    parser.add_argument('--mask-dir', dest='mask_dir', type=str, default='/data/receptor/human/crop/combined_final/mask/', help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--out-dir','-o', dest='out_dir', type=str, default='output', help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--scale-factors', dest='scale_factors_fn', type=str, default=None, help='json file with scaling and ordering info for each slab')
    parser.add_argument('--mri-gm', dest='srv_fn', type=str, default='/data/receptor/human/mri1_R_gm_bg_srv.nii.gz', help='mri gm super-resolution volume (srv)')
    parser.add_argument('--cortex-gm', dest='srv_cortex_fn', type=str, default='/data/receptor/human/mri1_gm_srv.nii.gz', help='mri gm super-resolution volume (srv)')
    parser.add_argument('--surf-dir', dest='surf_dir', type=str, default='/data/receptor/human/civet/mri1/surfaces/', help='surface directory')
    parser.add_argument('--autoradiograph-info', dest='autoradiograph_info_fn', type=str, default=None, help='csv file with section info for each autoradiograph')

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

                for resolution_itr, resolution in enumerate(args.resolution_list) :
                    cdict={} #current dictionary
                    cdict['brain']=brain        
                    cdict['hemi']=hemi        
                    cdict['slab']=slab        

                    # Directories
                    cdict['cur_out_dir']=f'{args.out_dir}/{brain}_{hemi}_{slab}/{resolution}mm/'
                    cdict['seg_dir']='{}/2_segment/'.format(cdict['cur_out_dir'])
                    cdict['srv_dir']=f'{args.out_dir}/{brain}_{hemi}_{slab}/srv/'
                    cdict['align_to_mri_dir']='{}/3_align_slab_to_mri/'.format(cdict['cur_out_dir'])
                    cdict['nl_2d_dir']='{}/4_nonlinear_2d'.format(cdict['cur_out_dir'])
                    
                    if resolution == args.resolution_list[0]:
                        cdict['init_align_dir']=f'{args.out_dir}/{brain}_{hemi}_{slab}/1_init_align/'
                        cdict['init_align_fn']=cdict['init_align_dir'] + f'/{brain}_{hemi}_{slab}_init_align.nii.gz'

                    # Filenames
                    cdict['seg_rsl_fn']='{}/{}_{}_{}_seg_{}mm.nii.gz'.format(cdict['seg_dir'],brain, hemi, slab, resolution)
                    cdict['srv_rsl_fn'] = cdict['srv_dir']+f'/{brain}_{hemi}_{slab}_mri_gm_{resolution}mm.nii.gz' 
                    cdict['srv_crop_rsl_fn'] = cdict['srv_dir']+f'/{brain}_{hemi}_{slab}_mri_gm_crop_{resolution}mm.nii.gz' 

                    #if resolution_itr <= max_3d_itr  :
                    cdict['rec_3d_rsl_fn'] = '{}/{}_{}_{}_{}mm_rec_space-mri.nii.gz'.format(cdict['align_to_mri_dir'],brain,hemi,slab,resolution)
                    cdict['srv_3d_rsl_fn'] = '{}/{}_{}_{}_{}mm_mri_gm_space-rec.nii.gz'.format(cdict['align_to_mri_dir'],brain,hemi,slab,resolution)
                    manual_tfm_dir = cdict['align_to_mri_dir']
                    cdict['manual_alignment_points'] = f'{manual_dir}/3d/{brain}_{hemi}_{slab}_points.txt'
                    cdict['manual_alignment_affine'] = f'{manual_dir}/3d/{brain}_{hemi}_{slab}_manual_affine.mat'
                    cdict['nl_3d_tfm_fn'] = f'{cdict["align_to_mri_dir"]}/{brain}_{hemi}_{slab}_rec_to_mri_{resolution}mm_SyN_CC_Composite.h5'
                    cdict['nl_3d_tfm_inv_fn'] = f'{cdict["align_to_mri_dir"]}/{brain}_{hemi}_{slab}_rec_to_mri_{resolution}mm_SyN_CC_InverseComposite.h5'

                    cdict['nl_2d_vol_fn'] = "{}/{}_{}_{}_nl_2d_{}mm.nii.gz".format(cdict['nl_2d_dir'] ,brain,hemi,slab,resolution) 
                    cdict['nl_2d_vol_cls_fn'] = "{}/{}_{}_{}_nl_2d_cls_{}mm.nii.gz".format(cdict['nl_2d_dir'],brain,hemi,slab,resolution) 
                    cdict['slab_info_fn'] = "{}/{}_{}_{}_{}_slab_info.csv".format(cdict['cur_out_dir'] ,brain,hemi,slab,resolution) 
                    cdict['srv_space_rec_fn'] = "{}/{}_{}_{}_srv_space-rec_{}mm.nii.gz".format(cdict['nl_2d_dir'], brain, hemi, slab, resolution)   
                    cdict['srv_iso_space_rec_fn'] = "{}/{}_{}_{}_srv_space-rec_{}mm_iso.nii.gz".format(cdict['nl_2d_dir'], brain, hemi, slab, resolution)   
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
    args.slabs =['1', '6', '2','5', '3', '4'] #FIXME shouldnt be hard coded

    if args.scale_factors_fn == None :
        args.scale_factors_fn=base_file_dir+'/scale_factors.json'

    args.manual_tfm_dir = base_file_dir + 'transforms/'

    if args.autoradiograph_info_fn == None :
        args.autoradiograph_info_fn=args.out_dir+'/autoradiograph_info_volume_order.csv'

    args.crop_dir = f'{args.out_dir}/0_crop'
    args.qc_dir = f'{args.out_dir}/6_quality_control/'
    os.makedirs(args.crop_dir,exist_ok=True)

    #args.landmark_src_dir=f'{manual_dir}/landmarks/'

    #args.landmark_dir=f'{args.out_dir}/landmarks/'
    #os.makedirs(args.landmark_dir, exist_ok=True)

    args.files_json = args.out_dir+"/reconstruction_files.json"
    args.manual_2d_dir=f'{manual_dir}/2d/'

    files = setup_files_json(args)

    return args, files 

def add_tfm_column(slab_df, init_tfm_csv, slab_tfm_csv) :
    tfm_df = pd.read_csv(init_tfm_csv)
    slab_df['init_tfm'] = [None] * slab_df.shape[0]

    for i, row in tfm_df.iterrows():
        slab_df['init_tfm'].loc[ slab_df['global_order'] == row['global_order'] ] = row['init_tfm']
   
    slab_df['tfm'] = slab_df['init_tfm']

    slab_df.to_csv(slab_tfm_csv)
    return slab_df

def create_directories(args, files, brain, hemi, resolution_list) :
    for slab in args.slabs :
        for resolution_itr, resolution in enumerate(resolution_list) :
            cfiles = files[brain][hemi][str(slab)][resolution] #Current files
            
            dirs_to_create=[ cfiles['cur_out_dir'], cfiles['seg_dir'], cfiles['srv_dir'],
                             cfiles['align_to_mri_dir'], cfiles['nl_2d_dir'] ]

            for dir_name in dirs_to_create : os.makedirs(dir_name, exist_ok=True)

def create_srv_volumes_for_next_slab(args,files, slab_list, resolution_list, resolution_list_3d, brain, hemi, slab, slab_index):

    not_last_slab = slab_index + 1 != len(slab_list)
    
    if not_last_slab  :
        resolution = resolution_list[-1]
        rec_3d_rsl_fn = files[brain][hemi][str(slab)][float(resolution)]['rec_3d_rsl_fn']
        # filenames for the next slab
        next_slab = args.slabs[slab_index+1]
        next_slab_files = files[brain][hemi][next_slab]

        # filename for cropped srv file for next slab
        stage_3_5_outputs = [ next_slab_files[float(r)]['srv_crop_rsl_fn']   for r in resolution_list ]
        print(stage_3_5_outputs)
        if run_stage( [], stage_3_5_outputs):
            print('\tNext Slab:', next_slab)
            print('\t\tStage 4.5 : Removing aligned slab from srv')

            crop_srv_rsl_fn = files[brain][hemi][str(int(slab))][float(resolution)]['srv_crop_rsl_fn']
            #DEBUG create_new_srv_volumes(rec_3d_rsl_fn, crop_srv_rsl_fn, stage_3_5_outputs, resolution_list_3d)
            create_new_srv_volumes(rec_3d_rsl_fn, args.srv_cortex_fn, stage_3_5_outputs, resolution_list_3d)

def surface_based_reconstruction(hemi_df, args, files, highest_resolution, slab_files_dict, interp_dir, brain, hemi, scale_factors, norm_df_csv=None) :
    ###
    ### Step 5 : Interpolate missing receptor densities using cortical surface mesh
    ###
    final_ligand_dict={}
    ligands = np.unique(hemi_df['ligand'])

    slabData = SlabReconstructionData(brain, hemi, args.slabs, ligands, args.depth_list, interp_dir, interp_dir +'/surfaces/', highest_resolution)
    
    for ligand, df_ligand in hemi_df.groupby(['ligand']):
        if ligand != 'cgp5' : continue
        print('\t\tLigand:', ligand)

        ligandSlabData = deepcopy(slabData)
        ligandSlabData.volumes = slabData.volumes[ligand] 
        ligandSlabData.cls = slabData.cls[ligand] 
        ligandSlabData.values_raw = slabData.values_raw[ligand] 
        ligandSlabData.values_interp = slabData.values_interp[ligand] 
        ligandSlabData.ligand = ligand
    
        create_thickened_volumes(interp_dir, slab_files_dict, hemi_df.loc[hemi_df['ligand']==ligand], slabData.n_depths, slabData.resolution, norm_df_csv=norm_df_csv)

        print('\tValidate reconstructed sections:', ligand)
        final_ligand_fn = args.out_dir + f'/reference_{highest_resolution}mm.nii.gz' 
        if not os.path.exists(final_ligand_fn) :
            prefilter_and_downsample(args.srv_cortex_fn, [float(highest_resolution)]*3, final_ligand_fn)
        validate_reconstructed_sections(final_ligand_fn, highest_resolution, args.n_depths+1, df_ligand, args.srv_cortex_fn, base_out_dir=args.out_dir,  clobber=False)
    
    exit(0)
    for ligand, df_ligand in hemi_df.groupby(['ligand']):
        #if ligand != 'cgp5' : continue
        print('\t\tLigand:', ligand)
        final_ligand_fn = surface_interpolation(ligandSlabData, df_ligand, slab_files_dict, args.srv_cortex_fn,  files[brain][hemi], scale_factors, input_surf_dir=args.surf_dir, n_vertices=args.n_vertices)
        final_ligand_dict[ligand] = final_ligand_fn

    return slabData, final_ligand_dict

def create_file_dict_output_resolution(files, brain, hemi, resolution_list):
    slab_files_dict={} 
    for slab, temp_slab_files_dict in files[brain][hemi].items() :
        nl_3d_tfm_exists = os.path.exists(temp_slab_files_dict[resolution_list[-1]]['nl_3d_tfm_fn'])
        nl_2d_vol_exists = os.path.exists(temp_slab_files_dict[resolution_list[-1]]['nl_2d_vol_fn'])
        if nl_3d_tfm_exists and nl_2d_vol_exists :
            slab_files_dict[int(slab)] = temp_slab_files_dict[resolution_list[-1]] 
        else : 
            print(f'Error: not including slab {slab} for interpolation (nl 3d tfm exists = {nl_3d_tfm_exists}, 2d nl vol exists = {nl_2d_vol_exists}) ')

    assert len(slab_files_dict.keys()) != 0 , print('No slabs to interpolate over')
    return slab_files_dict

def reconstruct_hemisphere(df, brain, hemi, args, files, resolution_list, max_resolution_3d=0.3):
    hemi_df = df.loc[ (df['mri']==brain) & (df['hemisphere']==hemi) ]
    hemi_df['nl_2d_rsl'] = ['empty'] * hemi_df.shape[0]
    hemi_df['nl_2d_cls_rsl'] = ['empty'] * hemi_df.shape[0]
    scale_factors = json.load(open(args.scale_factors_fn, 'r'))[brain][hemi]

    highest_resolution=resolution_list[-1]

    create_directories(args, files, brain, hemi, resolution_list)

    resolution_list_3d = [ float(resolution) for resolution in resolution_list + [max_resolution_3d] if float(resolution) >= max_resolution_3d ]
    
    dt = (1./args.n_depths)
    args.depth_list = np.round(np.arange(0, 1+dt/10, dt),3)
    #DEBUG args.depth_list = np.insert(args.depth_list,0, 0)

    ### Reconstruct slab
    for slab_index, slab in enumerate(args.slabs) :
        hemi_df = reconstruct_slab(hemi_df, brain, hemi, slab, slab_index, args, files, resolution_list, resolution_list_3d, max_resolution_3d=0.3)
   
    interp_dir=f'{args.out_dir}/5_surf_interp/'

    slab_files_dict = create_file_dict_output_resolution(files, brain, hemi, resolution_list)
    
    hemi_df, _ = validate_alignment(f'{args.qc_dir}/validate_alignment/', args.srv_cortex_fn, files[brain][hemi])

    hemi_df = hemi_df.loc[ hemi_df['align_dice'] > 0.5 ]
    #for (slab,align_dice), ddf in hemi_df.loc[hemi_df['ligand']=='cgp5'].groupby(['slab','align_dice']):
    #    print(ddf[['slab','align_dice']])

    if not args.no_surf : 
        print('\tSurface-based reconstruction') 
        slabData, final_ligand_dict = surface_based_reconstruction(hemi_df, args, files, highest_resolution, slab_files_dict, interp_dir, brain, hemi, scale_factors, norm_df_csv=args.norm_df_csv)
    ###
    ### 6. Quality Control
    ###
    df_list=[]
    for ligand, final_ligand_fn in final_ligand_dict.items() :
        df_ligand = hemi_df.loc[hemi_df['ligand']==ligand]
        max_resolution = resolution_list[-1]

        depth = args.depth_list[int((len(args.depth_list)+2)/2) ]

    for ligand, final_ligand_fn in final_ligand_dict.items() :
        ligand_csv_path = f'{interp_dir}/*{ligand}_{max_resolution}mm_l{args.n_depths+2}*{depth}_raw.csv'
        ligand_csv_list = glob(ligand_csv_path)
        if len(ligand_csv_list) > 0 : 
            ligand_csv = ligand_csv_list[0]
        else :
            print('Errpr: could not find ligand_csv', ligand_csv_path)
            exit(1)

        sphere_mesh_fn = glob(f'{interp_dir}/surfaces/surf_{max_resolution}mm_{depth}_sphere_rsl.npz')[0]
        cortex_mesh_fn = glob(f'{interp_dir}/surfaces/surf_{max_resolution}mm_{depth}_rsl.npz')[0]

        sphere_mesh_orig_fn = glob(f'{interp_dir}/surfaces/surf_{max_resolution}mm_0.0.surf.gii')[0]
        cortex_mesh_orig_fn = glob(f'{interp_dir}/surfaces/surf_{max_resolution}mm_0.0.sphere')[0]
        inflation_ratio = calculate_dist_of_sphere_inflation(cortex_mesh_orig_fn, sphere_mesh_orig_fn)
        
        sphere_resolution = float(max_resolution) * inflation_ratio
        tdf = validate_interpolation(ligand_csv, 
                            sphere_mesh_fn,
                            cortex_mesh_fn,  
                            args.qc_dir, 
                            max_resolution, 
                            ligand=ligand,
                            n_samples=10000,
                            clobber=False )
        df_list.append(tdf)
    df = pd.concat(df_list)

    out_r2_fn = f'{args.qc_dir}/interpolation_validation_r2.png'

    plot_r2(df, out_r2_fn)

from utils.mesh_utils import load_mesh_ext
from utils.utils import get_edges_from_faces

def calculate_dist_of_sphere_inflation(cortex_fn, sphere_fn):
    coord, faces = load_mesh_ext(cortex_fn)
    coord_sphere, faces_sphere = load_mesh_ext(sphere_fn)
    edges = get_edges_from_faces(faces)

    d0 = np.sqrt(np.sum(np.power(coord[edges[:,0]] - coord[edges[:,1]], 2),axis=1))
    d1 = np.sqrt(np.sum(np.power(coord_sphere[edges[:,0]] - coord_sphere[edges[:,1]], 2),axis=1))

    edge_ratios = d1 / d0

    ratio_average = np.mean(edge_ratios)
    ratio_std = np.std(edge_ratios)
    return ratio_average


def reconstruct_slab(hemi_df, brain, hemi, slab, slab_index, args, files, resolution_list, resolution_list_3d, max_resolution_3d=0.3):
    print('Slab:', slab)
    cdict = files[brain][hemi][str(slab)][float(resolution_list[0])]
    slab_df=hemi_df.loc[ hemi_df['slab'].astype(int) ==int(slab) ]
    init_align_fn = cdict['init_align_fn']
    init_align_dir = cdict['init_align_dir']
    init_tfm_csv =f'{init_align_dir}/{brain}_{hemi}_{slab}_final.csv'
    slab_tfm_csv =f'{init_align_dir}/{brain}_{hemi}_{slab}_slab_tfm.csv'

    ###  Step 1: Initial Alignment

    if (not os.path.exists( init_align_fn) or not os.path.exists(init_tfm_csv) or args.clobber) :
        print('\tInitial rigid inter-autoradiograph alignment')
        receptorRegister(brain, hemi, slab, init_align_fn, init_tfm_csv, init_align_dir, args.manual_2d_dir, slab_df, scale_factors_json=args.scale_factors_fn, clobber=args.clobber)
    
    if not os.path.exists(slab_tfm_csv): 
        slab_df = add_tfm_column(slab_df, init_tfm_csv,slab_tfm_csv)
    else : slab_df = pd.read_csv(slab_tfm_csv)
  
    #apply_transforms_to_landmarks(args.landmark_df, slab_df, args.landmark_dir, init_align_fn)

    ### Steps 2-4 : Multiresolution alignment
    cfiles=files[brain][hemi][str(slab)]
    multiresolution_outputs = []
    for file_name in ['seg_rsl_fn','nl_3d_tfm_fn','nl_3d_tfm_inv_fn','rec_3d_rsl_fn','srv_3d_rsl_fn','srv_iso_space_rec_fn','srv_space_rec_fn','nl_2d_vol_fn','nl_2d_vol_cls_fn', 'slab_info_fn', 'srv_crop_rsl_fn', 'srv_rsl_fn']:
        multiresolution_outputs += [ cfiles[float(resolution)][file_name] for resolution in resolution_list ]
        
    #multiresolution_outputs+=[cfiles[float(resolution_list[-1])]['slab_info_fn']]
    if run_stage([init_align_fn], multiresolution_outputs) or args.clobber : 
        print('\tMultiresolution Alignment')
        multiresolution_alignment( slab_df, hemi_df, brain, hemi, slab, slab_index, args, files, resolution_list, resolution_list_3d, init_align_fn, max_resolution_3d)

    print('\tCreate SRV volumes for next slab')
    create_srv_volumes_for_next_slab(args,files, args.slabs, resolution_list, resolution_list_3d, brain, hemi, slab, slab_index)
    ###
    ### Stage 4.5 : Create a new srv_rsl_fn file that removes the currently aligned slab
    ###
    #check that the current slab isn't the last one

    slab_info_fn = cfiles[float(resolution_list[-1])]['slab_info_fn'] #Current files
    slab_df = pd.read_csv(slab_info_fn, index_col=None)

    slab_idx = hemi_df['slab'].astype(int) == int(slab)

    assert np.sum(slab_idx) == slab_df.shape[0] , f'Error: no slab {slab} in {hemi_df["slab"].values}'
    hemi_df['nl_2d_rsl'].loc[ slab_idx ] = slab_df['nl_2d_rsl'].values
    hemi_df['nl_2d_cls_rsl'].loc[ slab_idx ] = slab_df['nl_2d_cls_rsl'].values

    return hemi_df

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

    args, files = setup_parameters(setup_argparse().parse_args() )
    
    args.resolution_list = [ float(r) for r in args.resolution_list ]

    #Process the base autoradiograph csv
    if not os.path.exists(args.autoradiograph_info_fn) : 
        calculate_section_order(args.autoradiograph_info_fn,  args.out_dir, in_df_fn=file_dir+os.sep+'autoradiograph_info.csv')

    df = pd.read_csv(args.autoradiograph_info_fn)
    
    ### Step 0 : Crop downsampled autoradiographs
    pytorch_model=f'{base_file_dir}/caps/Pytorch-UNet/MODEL.pth'
    #pytorch_model=''
    df = df.loc[ (df['hemisphere'] == 'R') & (df['mri'] == 'MR1' ) ] #FIXME, will need to be removed

    flip_axes_dict = {'caudal_to_rostral':(1,)}
    print('\tCropping')
    crop( args.crop_dir, args.mask_dir, df, args.scale_factors_fn, float(args.resolution_list[-1]), flip_axes_dict=flip_axes_dict,  pytorch_model=pytorch_model )
    
    print('\tFinished cropping') 
    #args.landmark_df = process_landmark_images(df, args.landmark_src_dir, args.landmark_dir, args.scale_factors_fn)

    #df = correct_batch_effects(df, args, n_samples=30, maxiter=10000, tolerance=1e-9)
    
    args.norm_df_csv = None
    
    for brain in args.brain :
        for hemi in args.hemi :                     
            print('Brain:',brain,'Hemisphere:', hemi)
            reconstruct_hemisphere(df, brain, hemi,  args, files, args.resolution_list)

