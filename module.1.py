from detectLines import train_model, apply_model

def tif2hdf5(dir_path, section_df, out_fn, clobber=False):
    '''
        Combine input images into an hdf5 volume
    '''
    if not os.path.exists(out_fn) or clobber :
        for row in section_df.iterrows() :



# Module 1 (per slab)
#   submodule 1.1:
#       IN: 1) raw tif
#       --> tif to nifti 
#       --> downsample
#       --> remove lines
#       --> crop
#       OUT: 1) raw.nii.gz 2) raw_rsl.nii.gz
#
#   submodule 1.2:
#       IN : raw_rsl.nii.gz
#       --> init reconstruction 
#       OUT: 1) tfm.h5 2) init_vol.nii.gz 3) init.h5
#
#   submodule 1.3:
#      IN : init_vol.nii.gz 
#       --> gm mask
#      OUT : gm_vol.nii.gz

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input','-i', dest='source', default='data/',  help='Directory with raw images')
    parser.add_argument('--output','-o', dest='output', default='output/', help='Directory name for outputs')
    parser.add_argument('--slab','-s', dest='slab', help='Brain slab')
    parser.add_argument('--hemi','-h', dest='hemi', help='Brain hemisphere')
    parser.add_argument('--brain','-b', dest='brain', help='Brain number')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    # Setup variables 
    #downsample
    scale_factors_json = check_file(args.source + os.sep + "scale_factors.json")
    z_mm = scale[self.brain_id][self.hemi][str(self.slab)]["size"]
    auto_affine = np.array([[z_mm/4164.,0,0,0],[0.0,z_mm/4164.,0,0],[0,0,1.0,0],[0,0,0,1]])
    input_raw = args.source + 'lin' 
    input_lin = args.source + 'raw'

    lines_removed_dir = args.output+"/lines_removed/"
    cropped_dir = args.output+"/crop/"
    downsampled_dir = args.output + '/downsampled/'
    if not os.path.exists(downsampled_dir) : os.makedirs(downsampled_dir)
    
    ### Step 1 : Put linearized and raw sections into HDF5 file
    tif2hdf5(lin_source_dir, lin_h5_fn)
    tif2hdf5(raw_source_dir, raw_h5_fn)


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

    ###########################################
    ### Step 1 : Downsample GM and Receptor, preserve sulci ###
    ###########################################
    if not os.path.exists(receptor_lines_removed_fn) or clobber :
        dwn_vol = np.zeros()
        for idx, receptor_fn in enumerate(.csv):
            img = imageio.imread(receptor_fn)
            if len(img.shape) == 3 : img = np.mean(img,axis=2)
            dwn_vol[:,idx,:] = nib.processing.resample_to_output(nib.Nifti1Image(img, auto_affine), .25, order=5).get_data()

    #Next step, inter-autoradiograph alignment

