import h5py as h5
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import imageio
import argparse
import os
import json
import nibabel as nib
import nibabel.processing
#from detectLines import train_model, apply_model

def check_file(fn):
    '''
        Verify that a file exists. If it does not return with error, otherwise return path.
    '''
    if not os.path.exists(fn) :
        print("Error could not find path for", fn)
        exit(ERROR_FILENOTFOUND)
    return fn

def read_image(fn):
    img = imageio.imread(fn)
    if len(img.shape) == 3 : img = np.mean(img,axis=2)
    return img

def downsample_image(img_fn) :
    print(img_fn)
    img = read_image(img_fn)
    img_nii = nib.Nifti1Image(img, auto_affine)
    img_dwn = nib.processing.resample_to_output( img_nii, .1, order=5).get_data()
    img_dwn = img_dwn.reshape(img_dwn.shape[0:2])
    return img_dwn


def downsample(df, dwn_fn, clobber=0) :
    
    img0 = downsample_image( df['lin_fn'].iloc[0] ) 
    if not os.path.exists(dwn_fn) or clobber > 0:
        n = df.shape[0]
        dwn_vol = h5.File(dwn_fn, "w")
        dwn_vol.create_dataset( 'data', (n, img0.shape[0], img0.shape[1]),  compression="gzip", dtype='float16')
        dwn_vol.create_dataset( 'flip', (n,),  compression="gzip", dtype='bool')
        dwn_vol['flip'][:] =False
    else : 
        dwn_vol = h5.File(dwn_fn, "r+")

    for i, (slab_i, row) in enumerate(df.iterrows()) :
        if np.max(dwn_vol['data'][i,:,:]) == 0 :
            dwn_img = downsample_image( row['lin_fn'] )
            if dwn_img.shape != img0.shape :
                dwn_img = dwn_img.T
                dwn_vol['flip'][i] = True
            dwn_vol['data'][i,:,:] = dwn_img 

    return dwn_vol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input','-i', dest='source', default='data/',  help='Directory with raw images')
    parser.add_argument('--csv','-c', dest='auto_info_fn', default='autoradiograph_info.csv',  help='csv file with autoradiograph information for each section')
    parser.add_argument('--output','-o', dest='output', default='output/', help='Directory name for outputs')
    parser.add_argument('--slab','-s', dest='slab',type=int, help='Brain slab')
    parser.add_argument('--hemi','-H', dest='hemi', help='Brain hemisphere')
    parser.add_argument('--brain','-b', dest='brain', help='Brain number')
    parser.add_argument('--clobber', dest='clobber', type=int, default=0, help='Clobber results')

    args = parser.parse_args()
    # Setup variables 
    scale_factors_json = check_file(args.source + os.sep + "scale_factors.json")
    scale = json.load(open(scale_factors_json,'r'))
    z_mm = scale[args.brain][args.hemi][str(args.slab)]["size"]
    auto_affine = np.array([[z_mm/4164.,0,0,0],[0.0,z_mm/4164.,0,0],[0,0,1.0,0],[0,0,0,1]])

    dwn_fn = '%s/brain-%s_hemi-%s_slab-%s_space-rec_100um_lin.h5.gz' % ( args.output, args.brain, args.hemi, args.slab  )
    #Create output directory
    if not os.path.exists(args.output) : os.makedirs(args.output)

    #Read .csv with autoradiograph information
    df = pd.read_csv(args.auto_info_fn)
    df = df.loc[ (df['hemisphere'] == args.hemi) & (df['mri'] == args.brain) & (df['slab']==args.slab) ]

    ###########################################
    ### Step 1 : Downsample GM and Receptor, preserve sulci ###
    ###########################################
    dwn_vol = downsample(df, dwn_fn, clobber=args.clobber) 

    '''
    if not os.path.exists(receptor_lines_removed_fn) or clobber :
        for y in range(dwn_vol.shape[1]) :
            ###################################
            ### Step 2 : Apply line removal ###
            ###################################
            if not os.path.exists(receptor_lines_removed_fn) or clobber :
                #Detect lines at 250um
                line_vol[:,y,:] = get_lines(dwn_vol[:,y:,], raw_files,max_model, raw_output_dir,  clobber)
        
    if not os.path.exists(receptor_lines_removed_fn) or clobber :
        for idx, receptor_fn in enumerate(.csv):
                #Remove lines at 20um
                remove_lines(line_files, raw_files, raw_output_dir, clobber)

    if not os.path.exists(receptor_lines_removed_fn) or clobber :
        for idx, receptor_fn in enumerate(.csv):
            ############################
            # Step 4 : GM Segmentation #
            ############################
            #At 20um resolution

            ##############################
            # Step 3: Automatic Cropping #
            ##############################
            #At 250um
            if not os.path.exists(receptor_cropped_fn) or clobber :
                crop_source_files(, cropped_dir, downsample_step=0.2, manual_only=True, no_frame=no_frame, ext='.png',clobber=False)
    '''


