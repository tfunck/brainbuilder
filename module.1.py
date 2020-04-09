
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
    parser.add_argument('--csv','-c', dest='autoradiograph_info_fn', default='autoradiograph_info.csv',  help='csv file with autoradiograph information for each section')
    parser.add_argument('--output','-o', dest='output', default='output/', help='Directory name for outputs')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    print("args")
    df = pd.read_csv(args.autoradiograph_info_fn)
    # Setup variables 
    slab = self.slab
    hemi = self.hemi
    brain = self.brain_id
    self.slab_output_path = self.slab_output_path 
    
    # Downsample

    ############################################
    ### Step 0: Train line detection network ###
    ############################################
    #if train-model :
        #python3 detectLines.py --epochs 5 --step 0.2 --train-source "test/" --train-output "line_detection_model/"

    ###################################
    ### Step 1 : Apply line removal ###
    ###################################
    #shell("python3 detectLines.py --step "+str(0.2)+"  --train-output line_detection_model/  --raw-source "+ raw_source+os.sep+hemi+"_slab_"+str(slab)+" --lin-source "+ lin_source+os.sep+hemi+"_slab_"+str(slab) +" --raw-output " + lines_output_dir+ " --ext .TIF") #--train-source \"test/\"
    #apply_model("line_detection_model/",source_raw_dir,source_lin_dir,lines_output_dir,0.2, clobber=False)

    ##############################
    # Step 2: Automatic Cropping #
    ##############################
    no_frame=False
    if brain+"/"+hemi+"_slab_"+str(slab) in dont_remove_lines : no_frame=True
    crop_source_files(self.slab_output_path+"/lines_removed/", self.crop_output_dir, downsample_step=0.2, manual_only=True, no_frame=no_frame, ext='.png',clobber=False)

    ##################################
    # Step 3 : Downsample Linearized #
    ##################################
    # Load scale factors for adjusting autoradiographs
    #scale_factors_json = check_file(args.source + os.sep + "scale_factors.json")
    scale_factors_json = self.args.source + os.sep + "scale_factors.json"

    if not os.path.exists(self.downsampled_dir) : os.makedirs(self.downsampled_dir)
    print(self.source_lin_dir, self.downsampled_dir)
    downsample_and_crop(self.source_lin_dir, self.downsampled_dir, self.crop_output_dir, self.auto_affine, clobber=self.args.clobber)

