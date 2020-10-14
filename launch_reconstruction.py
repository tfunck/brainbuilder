import os
import json
import re
import argparse
from utils.utils import shell, resample
from utils.ANTs import ANTs
from nibabel.processing import resample_to_output
from reconstruction.align_slab_to_mri import align_slab_to_mri
from reconstruction.receptor_segment import classifyReceptorSlices
from reconstruction.nonlinear_2d_alignment import receptor_2d_alignment, create_2d_sections, concatenate_sections_to_volume
from reconstruction.init_alignment import receptorRegister
from reconstruction.surface_interpolation import surface_interpolation
from reconstruction.crop import crop
import pandas as pd
import numpy as np
import nibabel as nib
global nl_2d_ext
nl_2d_ext='nl_2d.nii.gz'

global file_dir
base_file_dir, fn =os.path.split( os.path.abspath(__file__) )
file_dir = base_file_dir +os.sep +'section_numbers' +os.sep
def w2v(c, step, start):
    return np.round( (c-start)/step ).astype(int)

def adjust_slab(ll) :
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

    df['lin_fn'] = df['lin_fn'].apply( lambda x: os.path.splitext(os.path.basename(x))[0] ) 
    df['crop_fn']= df['lin_fn'].apply(lambda x: f'{out_dir}/{x}.nii.gz')

    df.to_csv(args.autoradiograph_info_fn)

def setup_argparse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--brains','-b', dest='brain', nargs='+', default=['MR1'], help='Brains to reconstruct. Default = run all.')
    parser.add_argument('--hemispheres', '--hemi', dest='hemi',default=['R'], nargs='+', help='Brains to reconstruct. Default = reconstruct all hemispheres.')
    parser.add_argument('--clobber', dest='clobber',default=False,action='store_true', help='Overwrite existing results')
    parser.add_argument('--slab','-s', dest='slab', nargs='+', default=[1], help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--chunk-perc','-u', dest='slab_chunk_perc', type=float, default=1., help='Subslab size (use with --nonlinear-only option) ')
    parser.add_argument('--chunk','-c', dest='slab_chunk_i', type=int, default=1, help='Subslab to align (use with --nonlinear-only option).')
    parser.add_argument('--nvertices', dest='n_vertices', type=int, default=327696, help='n vertices for mesh')
    parser.add_argument('--src-dir','-i', dest='src_dir', type=str, default='receptor_dwn', help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--out-dir','-o', dest='out_dir', type=str, default='reconstruction_output', help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--scale-factors', dest='scale_factors_fn', type=str, default=None, help='json file with scaling and ordering info for each slab')
    parser.add_argument('--mri-gm', dest='srv_fn', type=str, default=None, help='mri gm super-resolution volume (srv)')
    parser.add_argument('--surf-dir', dest='surf_dir', type=str, default='civet/mri1/surfaces/surfaces/', help='surface directory')
    parser.add_argument('--autoradiograph-info', dest='autoradiograph_info_fn', type=str, default=None, help='csv file with section info for each autoradiograph')
    parser.add_argument('--remote','-p', dest='remote', default=False, action='store_true',  help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--interpolation-only', dest='interpolation_only', default=False, action='store_true',  help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--resolution','-r', dest='resolution', type=str, default='3',  help='List of resolutions to process')

    return parser


def setup_files_json(args):
    if not os.path.exists(args.files_json) or args.clobber :
        files={}
        for brain in args.brain :
            files[brain]={}
            for hemi in args.hemi :
                files[brain][hemi]={}
                for slab in args.slab : 
                    files[brain][hemi][slab]={}
                    for resolution in resolution_list :
                        files[brain][hemi][slab][resolution] = {}
    else :
        files = json.load(open(args.files_json, 'r'))
        for brain in args.brain :
            try :
                files[brain]
            except KeyError :
                files[brain]={}

            for hemi in args.hemi :
                try :
                    files[brain][hemi]
                except KeyError:
                    files[brain][hemi]={}

                for slab in args.slab :
                    try :
                        files[brain][hemi][slab]
                    except KeyError :
                        files[brain][hemi][slab]={}

                    for resolution in resolution_list :
                        try:
                            files[brain][hemi][slab][resolution] 
                        except KeyError :
                            files[brain][hemi][slab][resolution] = {}
    return files

def setup_parameters(args) : 
    ###
    ### Parameters
    ###
    args.slab = adjust_slab(args.slab)

    if args.scale_factors_fn == None :
        args.scale_factors_fn=base_file_dir+'/scale_factors.json'

    if args.autoradiograph_info_fn == None :
        args.autoradiograph_info_fn=args.out_dir+'/autoradiograph_info_volume_order.csv'

    if args.srv_fn == None :
        args.srv_fn="srv/mri1_gm_bg_srv.nii.gz"

    args.crop_dir=f'{args.out_dir}/0_crop'
    os.makedirs(args.crop_dir,exist_ok=True)

    args.files_json=args.out_dir+"reconstruction_files.json"
    files = setup_files_json(args)

    return args, files 



def multiresolution_alignment(slab_df, hemi_df, brain, hemi, slab, args, files, resolution_list, init_align_fn):
    ### Iterate over progressively finer resolution
    for resolution_itr, resolution in enumerate(resolution_list) :
        print('Resolution',resolution)
        
        cur_out_dir=f'{args.out_dir}/{brain}_{hemi}_{slab}/{resolution}mm/'
        srv_rsl_fn = f'{args.out_dir}/{brain}_{hemi}_mri_gm_{resolution}mm.nii.gz' 
        #
        seg_dir=f'{cur_out_dir}/2_segment/'
        seg_rsl_fn=f'{seg_dir}/brain-{brain}_hemi-{hemi}_slab-{slab}_seg_{resolution}mm.nii.gz'
        #
        align_to_mri_dir = f'{cur_out_dir}/3_align_slab_to_mri/' 
        nl_3d_tfm_fn=f'{align_to_mri_dir}/rec_to_mri_Composite.h5'
        nl_3d_tfm_inv_fn=f'{align_to_mri_dir}/rec_to_mri_InverseComposite.h5'
        rec_3d_lin=f'{align_to_mri_dir}/{brain}_{hemi}_{slab}_rec_space-mri.nii.gz'
        srv_3d_lin=f'{align_to_mri_dir}/{brain}_{hemi}_{slab}_mri_gm_space-rec.nii.gz'
        #
        nl_2d_dir= f'{cur_out_dir}/4_nonlinear_2d'
        srv_base_rsl_crop_fn=f"{nl_2d_dir}/{brain}_{hemi}_{slab}_srv_rsl.nii.gz"
        nl_2d_vol = f"{nl_2d_dir}/{brain}_{hemi}_{slab}_{nl_2d_ext}"

        files[brain][hemi][slab][resolution]['nl_2d_dir']=nl_2d_dir
        files[brain][hemi][slab][resolution]['nl_2d_vol']=nl_2d_vol
        files[brain][hemi][slab][resolution]['srv_base_rsl_crop_fn'] = srv_base_rsl_crop_fn   
        files[brain][hemi][slab][resolution]['nl_3d_tfm'] = nl_3d_tfm_fn
        files[brain][hemi][slab][resolution]['nl_3d_tfm_inv'] = nl_3d_tfm_inv_fn
        json.dump(files,open(args.files_json,'w'))

        for dir_name in [cur_out_dir, align_to_mri_dir , seg_dir, nl_2d_dir ] :
            os.makedirs(dir_name, exist_ok=True)

        prev_resolution=resolution_list[resolution_itr-1]
        if resolution_itr > 0 : 
            align_fn = nl_2d_vol
        else : 
            align_fn = init_align_fn

        ###
        ### Step 1.5 : Downsample SRV to current resolution
        ###
        if not os.path.exists(srv_rsl_fn) or args.clobber :
            resample(nib.load(args.srv_fn), srv_rsl_fn, resolution)

        #Combine 2d sections from previous resolution level into a single volume
        if (resolution != resolution_list[0] and not os.path.exists(nl_2d_vol))  :
            last_nl_2d_dir = files[brain][hemi][slab][prev_resolution]['nl_2d_dir']
            concatenate_sections_to_volume(slab_df, init_align_fn, last_nl_2d_dir, nl_2d_vol)

        ###
        ### Step 2 : Autoradiograph segmentation
        ###
        print('\tStep 2: Autoradiograph segmentation')
        if (not os.path.exists(seg_rsl_fn) or args.clobber)  :
            classifyReceptorSlices(align_fn, seg_dir, seg_rsl_fn, rsl_dim=resolution)
        
        ###
        ### Step 3 : Align slabs to MRI
        ###
        print('\tStep 3: align slabs to MRI')
        dir_list = [nl_3d_tfm_fn, nl_3d_tfm_inv_fn, rec_3d_lin, srv_3d_lin]
        if False in [ os.path.exists(fn) for fn in dir_list ]  :
            #If removing aligned sections from srv set slabs= args.slab[slab_i:]
            align_slab_to_mri(seg_rsl_fn, srv_rsl_fn, slab, align_to_mri_dir, hemi_df, slabs, nl_3d_tfm_fn, nl_3d_tfm_inv_fn, rec_3d_lin, srv_3d_lin  )

        ###
        ### Step 4 : 2D alignment of receptor to resample MRI GM vol
        ###
        if not os.path.exists(srv_base_rsl_crop_fn) or args.clobber :
            shell(f'antsApplyTransforms -v 1 -d 3 -i {srv_rsl_fn} -r {init_align_fn} -t {nl_3d_tfm_inv_fn} -o {srv_base_rsl_crop_fn}', verbose=True)                   

        #create 2d sections that will be nonlinearly aliged in 2d
        create_2d_sections( slab_df, init_align_fn, srv_base_rsl_crop_fn, nl_2d_dir )
        
        #Check if necessary files exist to proceed to 2d nl alignment
        exit_early=False
        for fn in [ init_align_fn, srv_base_rsl_crop_fn, nl_2d_dir ] : 
            if not os.path.exists(fn) :
                print('Error: could not run 2d nonlinear alignment, missing', fn)
                exit_early=True
        if exit_early : exit(1)
            
        print('\tStep 4: 2d nl alignment')
       
        receptor_2d_alignment( slab_df, init_align_fn, srv_base_rsl_crop_fn, nl_2d_dir,  resolution, resolution_itr, batch_processing=args.remote)
    #Concatenate 2D nonlinear aligned sections into output volume
    if (resolution != resolution_list[0] and not os.path.exists(nl_2d_vol))  :
        concatenate_sections_to_volume(slab_df, init_align_fn, nl_2d_dir, nl_2d_vol)

def reconstruct_slab(slab_df, hemi_df, brain, hemi, slab, args, files, resolution_list):

        ###  Step 1: Initial Alignment
        print('\tInitial rigid inter-autoradiograph alignment')
        args.out_dir_1=f'{args.out_dir}/{brain}_{hemi}_{slab}/1_init_align/'
        init_align_fn=f'{args.out_dir_1}/brain-{brain}_hemi-{hemi}_slab-{slab}_init_align.nii.gz'
        files[brain][hemi][slab][resolution_list[0]]['init_align_fn'] = init_align_fn

        if (not os.path.exists( init_align_fn) or args.clobber) and not args.remote  :
            receptorRegister(brain,hemi,slab, init_align_fn, args.out_dir_1, slab_df, scale_factors_json=args.scale_factors_fn, clobber=args.clobber)

        multiresolution_alignment(slab_df, hemi_df, brain, hemi, slab, args,files, resolution_list, init_align_fn)

def reconstruct_hemisphere(df, brain, hemi,  args, files, resolution_list):

    
    hemi_df = df.loc[ (df['mri']==brain) & (df['hemisphere']==hemi) ]
    highest_resolution=resolution_list[-1]

    if not args.interpolation_only :
        ### Reconstruct slab
        for slab in args.slab :
            slab_df=df.loc[(df['hemisphere']==hemi) & (df['mri']==brain) & (df['slab']==int(slab)) ]
            reconstruct_slab(slab_df, hemi_df, brain, hemi, slab, args, files, resolution_list)

    ### Step 5 : Interpolate missing receptor densities using cortical surface mesh
    interp_dir=f'{args.out_dir}/5_surf_interp/'

    nl_tfm_list= []
    nl_2d_list = []
    for slab_dict in files[brain][hemi].values() :
        print( slab_dict[highest_resolution] )
        try :
            nl_2d_fn  = slab_dict[highest_resolution]['nl_2d_vol']
            nl_tfm_fn = slab_dict[highest_resolution]['nl_3d_tfm']
        except KeyError:
            continue

        if os.path.exists(nl_2d_fn) and os.path.exists(nl_tfm_fn) :
            nl_tfm_list.append(nl_tfm_fn)
            nl_2d_list.append(nl_2d_fn)
    
    # Surface interpolation
    if not args.remote or args.interpolation_only:
        surface_interpolation(nl_tfm_list,  nl_2d_list, interp_dir, brain, hemi, highest_resolution, hemi_df, args.srv_fn, surf_dir=args.surf_dir, n_vertices=args.n_vertices, n_depths=100)

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
    resolution_list = [ '3', '2', '1' , '0.5']#, '0.25']

    args, files = setup_parameters(setup_argparse().parse_args() )
    #Process the base autoradiograph csv
    if not os.path.exists(args.autoradiograph_info_fn) : 
        calculate_section_order(args.autoradiograph_info_fn, args.crop_dir, args.out_dir, in_df_fn=file_dir+os.sep+'autoradiograph_info.csv')

    df = pd.read_csv(args.autoradiograph_info_fn)

    ### Step 0 : Crop downsampled autoradiographs
    crop(args.src_dir,args.crop_dir, df,  remote=args.remote)

    for brain in args.brain :
        for hemi in args.hemi :                     
            reconstruct_hemisphere(df, brain, hemi,  args, files, resolution_list)

