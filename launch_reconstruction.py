import os
import json
import re
import argparse
from utils.utils import shell, resample
from utils.ANTs import ANTs
from nibabel.processing import resample_to_output
from reconstruction.align_slab_to_mri import align_slab_to_mri
from reconstruction.receptor_segment import classifyReceptorSlices
from reconstruction.nonlinear_2d_alignment import receptor_2d_alignment, create_2d_sections
from reconstruction.init_alignment import receptorRegister
from reconstruction.surface_interpolation import surface_interpolation
from reconstruction.crop import crop
import pandas as pd
import numpy as np
import nibabel as nib

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
    parser.add_argument('--src-dir','-i', dest='src_dir', type=str, default='receptor_dwn', help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--out-dir','-o', dest='out_dir', type=str, default='reconstruction_output', help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--scale-factors', dest='scale_factors_fn', type=str, default=None, help='json file with scaling and ordering info for each slab')
    parser.add_argument('--mri-gm', dest='srv_fn', type=str, default=None, help='mri gm super-resolution volume (srv)')
    parser.add_argument('--autoradiograph-info', dest='autoradiograph_info_fn', type=str, default=None, help='csv file with section info for each autoradiograph')
    parser.add_argument('--remote','-p', dest='remote', default=False, action='store_true',  help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--nonlinear-only', dest='nonlinear_only', default=False, action='store_true',  help='Slabs to reconstruct. Default = reconstruct all slabs.')
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

if __name__ == '__main__':
    resolution_list = [ '3', '2', '1' , '0.5', '0.25']

    args, files = setup_parameters(setup_argparse().parse_args() )

    #Process the base autoradiograph csv
    if not os.path.exists(args.autoradiograph_info_fn) : 
        calculate_section_order(args.autoradiograph_info_fn, args.crop_dir, args.out_dir, in_df_fn=file_dir+os.sep+'autoradiograph_info.csv')

    df = pd.read_csv(args.autoradiograph_info_fn)
    ### Step 0 :
    crop(args.src_dir,args.crop_dir, df,  remote=args.remote)

    for brain in args.brain :
        for hemi in args.hemi : 
            for slab_i, slab in enumerate(args.slab) : 
                ###
                ###  Step 1: Initial Alignment
                ###
                print('\tInitial rigid inter-autoradiograph alignment')
                args.out_dir_1=f'{args.out_dir}/{brain}_{hemi}_{slab}/1_init_align/'
                init_align_fn=f'{args.out_dir_1}/brain-{brain}_hemi-{hemi}_slab-{slab}_init_align.nii.gz'
                files[brain][hemi][slab][resolution_list[0]]['init_align_fn'] = init_align_fn
                    
                slab_df=df.loc[  (df['hemisphere']==hemi) & (df['mri']==brain) & (df['slab']==int(slab)) ]

                if (not os.path.exists( init_align_fn) or args.clobber) and not args.remote and not args.nonlinear_only :
                    receptorRegister(brain,hemi,slab, init_align_fn, args.out_dir_1, slab_df, scale_factors_json=args.scale_factors_fn, clobber=args.clobber)
               
                ### Iterate over progressively finer resolution
                for resolution_itr, resolution in enumerate(resolution_list) :
                    
                    print('Resolution',resolution)
                    
                    cur_out_dir=f'{args.out_dir}/{brain}_{hemi}_{slab}/{resolution}mm/'
                    os.makedirs(cur_out_dir,exist_ok=True)
                    lin_dir=f'{cur_out_dir}/3_align_slab_to_mri/'
                    files[brain][hemi][slab]['lin_dir'] = lin_dir
                    
                    ###
                    ### Step 1.5 : Downsample SRV to current resolution
                    ###
                    srv_rsl_fn = f'{cur_out_dir}/srv_{resolution}mm.nii.gz' 
                    if not os.path.exists(srv_rsl_fn) or args.clobber :
                        resample(nib.load(args.srv_fn), srv_rsl_fn, resolution)
     
                    srv_base_rsl_fn = f'{cur_out_dir}/srv_base_{resolution}mm.nii.gz' 
                    if not os.path.exists(srv_base_rsl_fn) or args.clobber :
                        resample(nib.load(srv_fn), srv_base_rsl_fn, resolution)

                    ###
                    ### Step 2 : Autoradiograph segmentation
                    ###
                    print('\tAutoradiograph segmentation')
                    if resolution_itr > 0 : align_fn = nl_2d_vol
                    else : align_fn = init_align_fn

                    seg_dir=f'{cur_out_dir}/2_segment/'
                    seg_rsl_fn=f'{seg_dir}/brain-{brain}_hemi-{hemi}_slab-{slab}_seg_{resolution}mm.nii.gz'
                    if (not os.path.exists(seg_rsl_fn) or args.clobber) and not args.nonlinear_only :
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

                    try :
                        files[brain][hemi][slab][resolution]['nl_3d_tfm']
                        files[brain][hemi][slab][resolution]['nl_3d_tfm_inv']
                    except KeyError :
                        nl_3d_tfm_fn, nl_3d_tfm_inv_fn, rec_3d_lin, srv_3d_lin = align_slab_to_mri(seg_rsl_fn, srv_rsl_fn, slab, align_to_mri_dir, hemi_df, args.slab[slab_i:], tfm  )
                        files[brain][hemi][slab][resolution]['nl_3d_tfm'] = nl_3d_tfm_fn
                        files[brain][hemi][slab][resolution]['nl_3d_tfm_inv'] = nl_3d_tfm_inv_fn

                    ###
                    ### Step 4 : 2D alignment of receptor to resample MRI GM vol
                    ###
                    nl_2d_dir= f'{cur_out_dir}/4_nonlinear_2d'
                    os.makedirs(nl_2d_dir, exist_ok=True)
                    srv_base_rsl_crop_fn=f"{nl_2d_dir}/{brain}_{hemi}_{slab}_srv_rsl.nii.gz"
                    files[brain][hemi][slab][resolution]['nl_2d_dir']=nl_2d_dir
                    files[brain][hemi][slab][resolution]['nl_2d_vol']=f"{nl_2d_dir}/{brain}_{hemi}_{slab}_nl_2d.nii.gz"
                    files[brain][hemi][slab][resolution]['srv_base_rsl_crop_fn'] = srv_base_rsl_crop_fn   

                    init_align_rsl_fn=f"{nl_2d_dir}/{brain}_{hemi}_{slab}_init_align_{resolution}mm.nii.gz"

                    df_fn=f"reconstruction_output/1_init_align/{brain}_{hemi}_{slab}/final/{brain}_{hemi}_{slab}_final.csv"

                    files[brain][hemi][slab][resolution]['init_align_rsl_fn']=init_align_rsl_fn
                    nl_2d_vol=files[brain][hemi][slab][resolution]['nl_2d_vol']

                    direction=json.load(open(args.scale_factors_fn,'r'))[brain][hemi][slab]["direction"]


                    json.dump(files,open(args.files_json,'w'))
                    
                    #create 2d sections that will be nonlinearly aliged in 2d
                    create_2d_sections( slab_df, init_align_fn, init_align_rsl_fn, srv_base_rsl_crop_fn, nl_2d_dir )
                    
                    if not os.path.exists(srv_base_rsl_crop_fn) or args.clobber :
                        shell(f'antsApplyTransforms -v 1 -d 3 -i {srv_base_rsl_fn} -r {init_align_fn} -t {nl_3d_tfm_inv_fn} -o {srv_base_rsl_crop_fn}', verbose=True)                   

                    if not os.path.exists( nl_2d_vol ) :
                        print('\tNL 2D volume:', nl_2d_vol, os.path.exists( nl_2d_vol ))
                        receptor_2d_alignment( slab_df, init_align_fn, init_align_rsl_fn, srv_base_rsl_crop_fn, nl_2d_dir,  resolution, resolution_itr, batch_processing=args.nonlinear_only, direction=direction)
                    
                        if args.nonlinear_only : exit(0)
               
                        concatenate_sections_to_volume(df, init_align_fn, output_dir, out_fn)
            ###
            ### Step 5 : Interpolate missing receptor densities using cortical surface mesh
            ###
            if not args.nonlinear_only :
                interp_dir=f'{args.out_dir}/5_surf_interp/'
                nl_2d_list = [ files[brain][hemi][i][resolution_list[-1]]['nl_2d_vol'] for i in args.slab ]
                nl_tfm_list = [ files[brain][hemi][i][resolution_list[-1]]['nl_3d_tfm_inv'] for i in args.slab ] 
                surface_interpolation(nl_tfm_list,  nl_2d_list, interp_dir, brain, hemi, resolution, df, n_depths=100)
