import os
import json
import numpy as np
import nibabel as nib
import argparse
from nibabel.processing import resample_from_to
from utils.utils import shell, downsample_and_crop
from sys import exit
from re import sub
from kmeans_vol import kmeans_vol
from glob import glob
from classifyReceptorSlices import classifyReceptorSlices
from receptorCrop import crop_source_files
from utils.utils import downsample_y
from receptorInterpolate import receptorInterpolate, receptorSliceIndicator
from receptorAdjustAlignment import receptorRegister
from detectLines import apply_model
from findMRIslab import createSRV

dont_remove_lines=["MR1/R_slab_6"]

class Slab():
    def __init__ (self, slab_raw_path, slab_lin_path, slab, brain_id, hemi, args) :
        self.hemi = hemi
        self.slab_raw_path = slab_raw_path
        self.slab_lin_path = slab_lin_path
        self.brain_id = brain_id
        self.slab_basename = os.path.basename(slab_lin_path)
        self.slab_output_path = args.output + os.sep + self.brain_id + os.sep + self.slab_basename
        self.slab = slab
        self.source_lin_dir = self.slab_lin_path+os.sep+self.brain_id + os.sep + self.slab_basename
        self.source_raw_dir = self.slab_raw_path+os.sep+self.brain_id + os.sep + self.slab_basename
        self.args=args 
        #Setup output directory
        if not os.path.exists(self.slab_output_path) : 
            os.makedirs(self.slab_output_path)

    def _reconstruct(self, args) :
        if args.run_preprocess or args.run_init_alignment or args.run_mri_to_receptor or args.run_receptor_interpolate :
            self._preprocess()
        elif args.run_init_alignment  or args.run_mri_to_receptor or args.run_receptor_interpolate:
            self._initial_alignment()
        elif args.run_mri_to_receptor  or args.run_receptor_interpolate :
            self._mri_to_receptor()
        elif args.run_receptor_interpolate :
            self._receptor_interpolate()


    def _preprocess(self):
        '''   
            Preprocess autoradiographs
        ''' 
        # Setup variables 
        slab = self.slab
        hemi = self.hemi
        brain = self.brain_id
        self.slab_output_path = self.slab_output_path 


        #Output directories
        lin_dwn_dir = self.slab_output_path+os.sep+'lin_dwn'+os.sep
        lines_output_dir = self.slab_output_path + "/lines_removed"
        crop_output_dir = self.slab_output_path + "/crop" 
        
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
        crop_source_files(self.slab_output_path+"/lines_removed/", crop_output_dir, downsample_step=0.2, manual_only=True, no_frame=no_frame, ext='.png',clobber=False)

        ##################################
        # Step 3 : Downsample Linearized #
        ##################################

        # Load scale factors for adjusting autoradiographs
        #scale_factors_json = check_file(args.source + os.sep + "scale_factors.json")
        scale_factors_json = self.args.source + os.sep + "scale_factors.json"
        with open(scale_factors_json) as f : scale=json.load(f)
        z_mm = scale[brain][hemi][str(slab)]["size"]

        affine = np.array([[0.02,0,0,0],[0.0,0.02,0,0],[0,0,z_mm/4164.,0],[0,0,0,1]])
        if not os.path.exists(lin_dwn_dir) : os.makedirs(lin_dwn_dir)


    def _initial_alignment(self) : 
        '''
            Initial alignment of autoradiographs
        '''
        # Setup variables
        brain = self.brain_id
        hemi = self.hemi
        slab = self.slab

        # Setup output
        lines_output_dir = self.slab_output_path + "/lines_removed"
        crop_output_dir = self.slab_output_path + "/crop" 
        reg_output_dir = self.slab_output_path + os.sep + "coregistration"
        volume_dir = self.slab_output_path + os.sep + "/volume/"
        classify_dir = self.slab_output_path + os.sep + "/classify/"
        lin_dwn_dir = self.slab_output_path + os.sep + 'lin_dwn' + os.sep

        ##################################
        ### Step 3 : Receptor Register ###
        ##################################
        print(" Step 3 : Receptor Register")

        slice_order_fn= "section_numbers.csv"
        tiers_str = "flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg;sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18;oxot;epib"
        clobber=False
        unaligned_rec_fn =reg_output_dir +os.sep+"/vol_0.nii.gz"
        rec_fn =reg_output_dir +os.sep+"/vol_final.nii.gz"
        rec_df_fn=reg_output_dir +os.sep+"/autoradiograph_info.csv"
        transforms_fn=reg_output_dir +os.sep+"/transforms.json"
        if not os.path.exists(rec_fn) or clobber :
            receptorRegister(lin_dwn_dir, reg_output_dir, tiers_str, rec_df_fn, ext=".nii.gz",  clobber=False)

        ####################################
        # Step 5. Classify receptor volume #
        ####################################
        print("Step 5. Classify receptor volume")
        out_file="vol_cls_"+str(slab)+".nii.gz"
        classifyReceptorSlices(rec_fn, classify_dir, out_file, 5, False)
       
        ################################
        # Step 6. Resample cls and srv #
        ################################
        print("Step 6. Resample cls and srv")
        cls_fn = classify_dir + os.sep + "vol_cls_"+str(slab)+".nii.gz"
        cls_iso_dwn_fn = classify_dir + os.sep + "vol_cls_"+str(slab)+"_250um.nii.gz"
        if not os.path.exists(cls_iso_dwn_fn):
            downsample_y(cls_fn, cls_iso_dwn_fn, 0.25 )

        
    def _mri_to_receptor(self):
        '''                 
            Transform MRI to Receptor Volume 
        '''
        # Setup input variables
        brain = self.brain_id
        hemi = self.hemi
        slab = self.slab
            
        # Setup directories
        lines_output_dir = self.slab_output_path+"/lines_removed"
        crop_output_dir = self.slab_output_path+"/crop" 
        reg_output_dir = self.slab_output_path + os.sep + "coregistration"
        volume_dir = self.slab_output_path +os.sep+"/volume/"
        classify_dir = self.slab_output_path +os.sep+"/classify/"
        nonlinear_dir = self.slab_output_path+os.sep+"nonlinear/"

        #attenuation_file="mr1_attenuation_map.nii.gz"


        ###############################################
        ### Step 7: Transform MRI to receptor space ###
        ###############################################
        print("Step 7: Transform MRI to receptor space")
        srv = "srv/mri1_gm_bg_srv_slab-"+str(slab)+".nii.gz"
        srv2cls_base=nonlinear_dir+os.sep+"transform_"+str(slab)+"_"
        srv2cls=nonlinear_dir+os.sep+"warp_"+str(slab)+"_Composite.h5"
        cls2srv=nonlinear_dir+os.sep+"warp_"+str(slab)+"_InverseComposite.h5"
        srv_rsl=nonlinear_dir+os.sep+"srv_space-rec-"+str(slab)+"_lin_rsl.nii.gz"
        output_file=nonlinear_dir+os.sep+"warped_"+str(slab)+".nii"
        output_file_inverse=nonlinear_dir+os.sep+"warped_inverse_"+str(slab)+".nii"
        
        cls_fn = classify_dir + os.sep + "vol_cls_"+str(slab)+".nii.gz"
        cls_iso_dwn_fn = classify_dir + os.sep + "vol_cls_"+str(slab)+"_250um.nii.gz"

        if not os.path.exists(srv2cls) or not os.path.exists(cls2srv) : 
            if not os.path.exists(nonlinear_dir):
                os.makedirs(nonlinear_dir)
            
            #ANTs Registration
            cmdline="antsRegistration --verbose 1 --float --collapse-output-transforms 1 --dimensionality 3 --initial-moving-transform [ "+cls_iso_dwn_fn+", "+srv+", 1 ] --initialize-transforms-per-stage 0 --interpolation Linear"
            cmdline+=" --transform Rigid[ 0.1 ] --metric Mattes[ "+cls_iso_dwn_fn+", "+srv+", 1, 32, Regular, 0.3 ] --convergence [ 1000x500x500, 1e-08, 10 ] --smoothing-sigmas 8.0x4.0x3.0vox --shrink-factors 8x4x3 --use-estimate-learning-rate-once 1 --use-histogram-matching 0 "
            #cmdline += " --transform Affine[ 0.1 ] --metric Mattes[ "+cls_iso_dwn_fn+", "+srv+", 1, 32, Regular, 0.3 ] --convergence [ 1000x500x500 , 1e-08, 10 ] --smoothing-sigmas 8.0x4.0x2.0vox --shrink-factors 8x4x3 --use-estimate-learning-rate-once 1 --use-histogram-matching 0 "
            #cmdline += " --transform SyN[ 0.1, 3.0, 0.0] --metric Mattes[ "+cls_iso_dwn_fn+", "+srv+", 0.5, 64, None ]  --convergence [ 2000x1500x10000x500, 1e-6,10 ] --smoothing-sigmas 8.0x6.0x4.0x2.0vox --shrink-factors 8x4x2x1  --winsorize-image-intensities [ 0.005, 0.995 ]  --write-composite-transform 1 "
            cmdline += "--output [ "+nonlinear_dir+"warp_"+str(slab)+"_ ,"+output_file+","+output_file_inverse+"]"
            #1500x1000x500x250
            print(cmdline)
            shell(cmdline)

        if not os.path.exists(srv_rsl) :
            cmdline=" ".join(["antsApplyTransforms -n Linear -d 3 -i",srv,"-r",cls_fn,"-t",srv2cls,"-o",srv_rsl])
            print(cmdline)
            shell(cmdline)            


    def _receptor_interpolate(self):
        ############################################################
        ### 8. Interpolate missing slices for each receptor type ###
        ############################################################
        print("Step 8. Interpolate missing slices for each receptor type")
        ligands=["flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg,sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18,oxot,epib"]

        unaligned_rec_fn =reg_output_dir +os.sep+"/vol_0.nii.gz"
        rec_fn =reg_output_dir +os.sep+"/vol_final.nii.gz"
        rec_df_fn=reg_output_dir +os.sep+"/autoradiograph_info.csv"
        transforms_fn=reg_output_dir +os.sep+"/transforms.json"

        for ligand in ligands_to_run :
            ligand_dir = self.slab_output_path + os.sep + "ligand" + os.sep + ligand + os.sep + 'slab-'+str(slab)
            print("\tLigand:",ligand)
            receptorVolume = ligand_dir + os.sep + ligand + ".nii.gz"
            rec_slice_fn = ligand_dir + os.sep + ligand + "_slice_indicator.nii.gz"
            # lin --> lin_dwn --> crop --> 2D rigid (T1) --> 3D non-linear (T2) --> 2D non-linear (T3) [--> 2D non-linear (T4)]
            # lin_mri_space =  T2^-1 x [T4 x] T3 x T1 x lin_dwn 
            if not os.path.exists(receptorVolume) or clobber : 
                receptorInterpolate(i,rec_fn, srv_rsl, cls_fn, ligand_dir, ligand, rec_df_fn, transforms_fn, clobber=False)
            
            if not os.path.exists( rec_slice_fn  ) or clobber :
                receptorSliceIndicator( rec_df_fn, ligand, receptorVolume, 3, rec_slice_fn )
            
            ######################################################
            ### 9. Transform Linearized tif files to MRI space ###
            ######################################################
            print("9. Transform Linearized tif files to MRI space")
            final_dir = self.slab_output_path + os.sep + "final"
            if not os.path.exists(self.slab_output_path+'/final') : os.makedirs(self.slab_output_path+'/final')
            receptorVolume_mni_space = final_dir + os.sep + ligand + "_space-mni.nii.gz"
            if not os.path.exists(receptorVolume_mni_space) or clobber :
                cmdline=" ".join(["antsApplyTransforms -d 3 -n Linear  -i",receptorVolume,"-r",srv,"-t",cls2srv,"-o",receptorVolume_mni_space])
                print(cmdline)
                shell(cmdline)

            #receptorVolumeRsl = final_dir + os.sep + ligand + "_space-mni_500um.nii.gz"
            size=0.5
            final_base_fn = ligand + "_space-mni_"+str(size)+"mm"
            receptorVolumeRsl = final_dir + os.sep +final_base_fn+ ".nii.gz"
            if not os.path.exists(receptorVolumeRsl) or clobber : 
                img=nib.load(receptorVolume_mni_space)
                nib.processing.resample_to_output(nib.Nifti1Image(img.get_data(), img.affine), size, order=5).to_filename(receptorVolumeRsl)
                del img
                
            receptorVolume_final_kmeans = final_dir + os.sep + final_base_fn+"_segmented.nii.gz"
            receptorVolume_final_kmeans_rsl = final_dir + os.sep + final_base_fn+"_segmented_rsl.nii.gz"
            receptorVolume_final_dat = final_dir + os.sep + final_base_fn+"_segmented_rsl.dat"
            receptorVolume_final_kmeans_anlz = final_dir + os.sep + final_base_fn + "_segmented_rsl.img"
            if not os.path.exists( receptorVolume_final_kmeans ) or not os.path.exists(receptorVolume_final_dat) or clobber :
                kmeans_vol(receptorVolumeRsl, 200,receptorVolume_final_dat, receptorVolume_final_kmeans)                
            #Resample segmented receptor volume to attenuation map, so that they overlap
            #This makes it easier to define offset w.r.t center of scanner FOV for both volumes in Gate
            if not os.path.exists( receptorVolume_final_kmeans_rsl ) or clobber :
                img1 = nib.load(receptorVolume_final_kmeans)
                vol1 = img1.get_data()
                vol2 = nib.load(attenuation_file).get_data()
                print(vol1.shape)
                print(vol1.dtype.name)
                print(vol2.shape)
                print(vol2.dtype.name)
                rsl = resample_from_to( nib.load(receptorVolume_final_kmeans) , nib.load(attenuation_file) )
                rsl.set_data_dtype(img1.get_data_dtype())
                rsl.to_filename(receptorVolume_final_kmeans_rsl)
                shell("medcon -c anlz -f "+ receptorVolume_final_kmeans + ' -o '+receptorVolume_final_kmeans_anlz)
