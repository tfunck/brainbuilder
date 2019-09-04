import os
import json
import numpy as np
import nibabel as nib
import argparse
import pandas as pd
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
from ANTs import ANTs

dont_remove_lines=["MR1/R_slab_6"]

class Slab():
    def __init__ (self, slab_raw_path, slab_lin_path, slab, brain_id, hemi, args) :
        self.hemi = hemi
        self.slab_raw_path = slab_raw_path
        self.slab_lin_path = slab_lin_path
        self.brain_id = brain_id
        self.slab_basename = os.path.basename(slab_lin_path)
        #self.brain_output_path = args.output + os.sep + self.brain_id + os.sep
        self.slab_output_path = args.output + os.sep + self.brain_id + os.sep + self.slab_basename
        self.slab = slab
        self.source_lin_dir = self.slab_lin_path #+os.sep+self.brain_id + os.sep + self.slab_basename
        self.source_raw_dir = self.slab_raw_path #+os.sep+self.brain_id + os.sep + self.slab_basename
        self.clobber = args.clobber
        self.args=args 
        #Setup output directory
        if not os.path.exists(self.slab_output_path) : 
            os.makedirs(self.slab_output_path)

        # Setup directories
        self.lines_output_dir = self.slab_output_path+"/lines_removed"
        self.crop_output_dir = self.slab_output_path+"/crop" 
        self.reg_output_dir = self.slab_output_path + os.sep + "coregistration"
        self.classify_dir = self.slab_output_path +os.sep+"/classify/"
        self.mri_to_receptor_dir = self.slab_output_path+os.sep+"mri_to_receptor/"
        self.downsampled_dir = self.slab_output_path+os.sep+'downsampled'+os.sep
        self.rec_fn = self.reg_output_dir +os.sep+"/vol_final.nii.gz"

        with open(args.scale_factors_json) as f : scale=json.load(f)
        z_mm = scale[self.brain_id][self.hemi][str(self.slab)]["size"]
        self.auto_affine = np.array([[z_mm/4164.,0,0,0],[0.0,z_mm/4164.,0,0],[0,0,1.0,0],[0,0,0,1]])

        self.srv_rigid_rsl=self.mri_to_receptor_dir+os.sep+"srv_space-rec-"+str(self.slab)+"_rigid_rsl.nii.gz"
        self.srv_affine_rsl=self.mri_to_receptor_dir+os.sep+"srv_space-rec-"+str(self.slab)+"_affine_rsl.nii.gz"
        self.srv_rsl=self.mri_to_receptor_dir+os.sep+"srv_space-rec-"+str(self.slab)+"_nl_rsl.nii.gz"
        self.srv2cls_prefix=self.mri_to_receptor_dir+os.sep+"transform_"+str(slab)+"_"

        self.cls_fn = self.classify_dir + os.sep + "vol_cls_"+str(slab)+".nii.gz"
        
        self.srv2cls = self.srv2cls_prefix + 'Composite.h5'
        self.cls2srv = self.srv2cls_prefix + 'InverseComposite.h5'

    def _init_reconstruct(self, args) :
        if args.run_preprocess : #or args.run_init_alignment or args.run_mri_to_receptor or args.run_receptor_interpolate :
            print("Preprocess autoradiographs")
            self._preprocess()
        
        if args.run_init_alignment : #  or args.run_mri_to_receptor or args.run_receptor_interpolate:
            print("Initial Alignment")
            self._initial_alignment()
        
    def _global_reconstruct(self, args, slabs_srv_dict) :
        

        if args.run_mri_to_receptor  or args.run_receptor_interpolate :
            print("MRI to Receptor")
            self._mri_to_receptor(slabs_srv_dict)

        if args.run_receptor_interpolate :
            self._receptor_interpolate(args)


    def _preprocess(self):
        '''   
            Preprocess autoradiographs
        ''' 
        # Setup variables 
        slab = self.slab
        hemi = self.hemi
        brain = self.brain_id
        self.slab_output_path = self.slab_output_path 
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


    def _initial_alignment(self) : 
        '''
            Initial alignment of autoradiographs
        '''
        # Setup variables
        brain = self.brain_id
        hemi = self.hemi
        slab = self.slab

        ##################################
        ### Step 3 : Receptor Register ###
        ##################################
        print(" Step 3 : Receptor Register")
        slice_order_fn= "section_numbers.csv"
        tiers_str = "flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg;sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18;oxot;epib"
        unaligned_rec_fn =self.reg_output_dir +os.sep+"/vol_0.nii.gz"
        rec_fn = self.rec_fn
        rec_df_fn=self.reg_output_dir +os.sep+"/autoradiograph_info.csv"
        transforms_fn=self.reg_output_dir +os.sep+"/transforms.json"
        if not os.path.exists(rec_fn) or self.args.clobber :
            receptorRegister(self.downsampled_dir, self.reg_output_dir, tiers_str, rec_df_fn, ext=".nii.gz", n_epochs=self.args.init_align_epochs, clobber=self.args.clobber)

        ####################################
        # Step 5. Classify receptor volume #
        ####################################
        print("Step 5. Classify receptor volume")
        out_file="vol_cls_"+str(slab)+".nii.gz"
        classifyReceptorSlices(rec_fn, self.classify_dir, out_file, 5, self.args.clobber)
       
        ################################
        # Step 6. Resample cls and srv #
        ################################
        print("Step 6. Resample cls and srv")
        cls_iso_dwn_fn = self.classify_dir + os.sep + "vol_cls_"+str(slab)+"_250um.nii.gz"
        if not os.path.exists(cls_iso_dwn_fn) or self.args.clobber:
            downsample_y(self.cls_fn, cls_iso_dwn_fn, 0.25, clobber=self.args.clobber )

        
    def _mri_to_receptor(self, slab_srv_dict):
        '''                 
            Transform MRI to Receptor Volume 
        '''
        print("\nMRI to Receptor Volume\n")
        # Setup input variables
        brain = self.brain_id
        hemi = self.hemi
        slab = self.slab
        clobber = self.clobber
        self.srv = slab_srv_dict["fixed"].loc[ slab_srv_dict["slab"] == float(slab) ].values[0] 
        self.init_tfm = slab_srv_dict["tfm"].loc[ slab_srv_dict["slab"] == float(slab) ].values[0]

        #attenuation_file="mr1_attenuation_map.nii.gz"
        ###############################################
        ### Step 7: Transform MRI to receptor space ###
        ###############################################
        print("Step 7: Transform MRI to receptor space")

        syn_file=self.mri_to_receptor_dir+os.sep+"srv_syn_space-rec_"+str(slab)+".nii.gz"
        syn_file_inverse=self.mri_to_receptor_dir+os.sep+"cls_syn_space-mni_"+str(slab)+".nii.gz"
        
        affine_file=self.mri_to_receptor_dir+os.sep+"affine_"+str(slab)+".nii.gz"
        affine_file_inverse=self.mri_to_receptor_dir+os.sep+"affine_inverse_"+str(slab)+".nii.gz"

        cls_iso_dwn_fn = self.classify_dir + os.sep + "vol_cls_"+str(slab)+"_250um.nii.gz"
        
        def srv2cls(outDir, tfm_prefix, tfm_fn, fixed_fn, moving_fn, moving_rsl_fn, moving_rsl_fn_inverse, out_fn, init_tfm ) :
            print("\nANTs Command Line")
            if not os.path.exists(moving_rsl_fn_inverse) or not os.path.exists(moving_rsl_fn) or clobber : 
                if not os.path.exists(self.mri_to_receptor_dir):
                    os.makedirs(self.mri_to_receptor_dir)
             
                ANTs(outDir, self.srv2cls_prefix, fixed_fn, moving_fn, moving_rsl_fn, moving_rsl_fn_inverse, iterations=['500x250x100x10', '1000x500x300','3000x2000x1000x500x100'], base_shrink_factor=1, radius=64, metric="GC", verbose=1, clobber=1, init_tfm=self.init_tfm, init_inverse=True,  exit_on_failure=True)

            if (not os.path.exists(out_fn) or clobber ) and moving_rsl_fn != None:
                img = nib.load(moving_rsl_fn)
                rsl = nib.load(self.cls_fn)
                resample_from_to(img, rsl, order=0).to_filename(out_fn)
                print("Wrote:", out_fn)
        
        fixed_fn = cls_iso_dwn_fn
        moving_fn = self.srv
        moving_rsl_fn = self.mri_to_receptor_dir+os.sep+"syn_"+str(slab)+".nii.gz"
        moving_rsl_fn_inverse = self.mri_to_receptor_dir+os.sep+"syn_inverse_"+str(slab)+".nii.gz"
        out_fn = cls_iso_dwn_fn

        print("fixed", fixed_fn)
        print("moving_rsl", moving_rsl_fn)
        print("moving", moving_fn)
        srv2cls(self.mri_to_receptor_dir, self.srv2cls_prefix, self.srv2cls_prefix, fixed_fn, moving_fn, moving_rsl_fn, moving_rsl_fn_inverse, self.srv_rsl , self.init_tfm )

    def _receptor_interpolate(self,args):
        ############################################################
        ### 8. Interpolate missing slices for each receptor type ###
        ############################################################
        print("Step 8. Interpolate missing slices for each receptor type")
        ligands=["flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg,sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18,oxot,epib"]

        def write_ligand_volume(ligand, df_fn, rec_fn, out_fn, w=0) :
            df = pd.read_csv(df_fn)
            df = df.loc[ df["ligand"] == ligand ]
            rec = nib.load(rec_fn)
            recVolume = rec.get_data()
            outVolume=np.zeros(rec.shape)
            for i, row in df.iterrows() :
                y = row["volume_order"]

                outVolume[:,int(y),:] = recVolume[ :, int(y), : ]

                for j in range( max(0, y-w), min(y+w, recVolume.shape[1]) ) :
                    outVolume[ :, j, : ] = outVolume[ :, int(y), : ]

            nib.Nifti1Image(outVolume, rec.affine).to_filename(out_fn)

        unaligned_rec_fn =self.reg_output_dir +os.sep+"/vol_0.nii.gz"
        rec_fn =self.reg_output_dir +os.sep+"/vol_final.nii.gz"
        rec_df_fn=self.reg_output_dir +os.sep+"/autoradiograph_info.csv"
        transforms_fn=self.reg_output_dir +os.sep+"/transforms.json"

        for ligand in self.args.ligands_to_run :
            ligand_dir = self.slab_output_path + os.sep + "ligand" + os.sep + ligand 
            print("\tLigand:",ligand)
            receptorVolume = ligand_dir + os.sep + ligand + ".nii.gz"
            rec_slice_fn = ligand_dir + os.sep + ligand + "_slice_indicator.nii.gz"
            ligand_interp_fn = ligand_dir+os.sep+ligand+".nii.gz"
            ligand_no_interp_fn = ligand_dir+os.sep+ligand+"_no_interp.nii.gz"
            ligand_init_ligand_fn = ligand_dir+os.sep+ligand+"_init.nii.gz"
            # lin --> downsampled --> crop --> 2D rigid (T1) --> 3D non-linear (T2) --> 2D non-linear (T3) [--> 2D non-linear (T4)]
            # lin_mri_space =  T2^-1 x [T4 x] T3 x T1 x downsampled 
            print(receptorVolume)
            if not os.path.exists(receptorVolume) or self.args.clobber or self.args.validation : 
                receptorInterpolate(self.slab,ligand_interp_fn,rec_fn, self.srv_rsl, self.cls_fn, ligand_dir, ligand, rec_df_fn, transforms_fn, tfm_type_2d=args.tfm_type_2d, clobber=self.args.clobber)
            
            write_ligand_volume(ligand, rec_df_fn, ligand_interp_fn,ligand_no_interp_fn , 10 )
            write_ligand_volume(ligand, rec_df_fn, rec_fn, ligand_init_ligand_fn )

            if not os.path.exists( rec_slice_fn  ) or self.args.clobber :
                receptorSliceIndicator( rec_df_fn, ligand, receptorVolume, 3, rec_slice_fn )
            
            ######################################################
            ### 9. Transform Linearized tif files to MRI space ###
            ######################################################
            print("9. Transform Linearized tif files to MRI space")
            final_dir = self.slab_output_path + os.sep + "final"
            if not os.path.exists(self.slab_output_path+'/final') : os.makedirs(self.slab_output_path+'/final')
            receptorVolume_mni_space = final_dir + os.sep + ligand + "_space-mni.nii.gz"
            receptorVolume_no_interp_mni_space = final_dir + os.sep + ligand + "_no-interp_space-mni.nii.gz"

            if not os.path.exists(receptorVolume_no_interp_mni_space) or self.args.clobber :
                cmdline=" ".join(["antsApplyTransforms -d 3 -n BSpline[3] -i",ligand_no_interp_fn,"-r",self.srv,"-t",self.cls2srv,"-o",receptorVolume_no_interp_mni_space])
                print(cmdline)
                shell(cmdline)

            print( receptorVolume_mni_space )
            if not os.path.exists(receptorVolume_mni_space) or self.args.clobber :
                cmdline=" ".join(["antsApplyTransforms -d 3 -n BSpline[3] -i",receptorVolume,"-r",self.srv,"-t",self.cls2srv,"-o",receptorVolume_mni_space])
                print(cmdline)
                shell(cmdline)

            receptorVolumeRsl = final_dir + os.sep + ligand + "_space-mni_500um.nii.gz"
            size=0.5
            final_base_fn = ligand + "_space-mni_"+str(size)+"mm"
            if not os.path.exists(receptorVolumeRsl) or self.args.clobber : 
                img = nib.load(receptorVolume_mni_space)
                print("Writing:", receptorVolumeRsl)
                nib.processing.resample_to_output(nib.Nifti1Image(img.get_data(), img.affine), size, order=5).to_filename(receptorVolumeRsl)
                del img
                
            #receptorVolume_final_kmeans = final_dir + os.sep + final_base_fn+"_segmented.nii.gz"
            #receptorVolume_final_kmeans_rsl = final_dir + os.sep + final_base_fn+"_segmented_rsl.nii.gz"
            #receptorVolume_final_dat = final_dir + os.sep + final_base_fn+"_segmented_rsl.dat"
            #receptorVolume_final_kmeans_anlz = final_dir + os.sep + final_base_fn + "_segmented_rsl.img"
            #if not os.path.exists( receptorVolume_final_kmeans ) or not os.path.exists(receptorVolume_final_dat) or self.args.clobber :
            #    kmeans_vol(receptorVolumeRsl, 200,receptorVolume_final_dat, receptorVolume_final_kmeans)                
            #Resample segmented receptor volume to attenuation map, so that they overlap
            #This makes it easier to define offset w.r.t center of scanner FOV for both volumes in Gate
            #if not os.path.exists( receptorVolume_final_kmeans_rsl ) or  self.args.clobber :
            #    img1 = nib.load(receptorVolume_final_kmeans)
            #    vol1 = img1.get_data()
            #    vol2 = nib.load(attenuation_file).get_data()
            #    print(vol1.shape)
            #    print(vol1.dtype.name)
            #    print(vol2.shape)
            #    print(vol2.dtype.name)
            #    rsl = resample_from_to( nib.load(receptorVolume_final_kmeans) , nib.load(attenuation_file) )
            #    rsl.set_data_dtype(img1.get_data_dtype())
            #    rsl.to_filename(receptorVolume_final_kmeans_rsl)
            #    shell("medcon -c anlz -f "+ receptorVolume_final_kmeans + ' -o '+receptorVolume_final_kmeans_anlz)
