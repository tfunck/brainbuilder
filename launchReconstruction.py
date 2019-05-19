import os
import json
import numpy as np
import nibabel as nib
from scipy.ndimage.filters import gaussian_filter
from utils.utils import shell
from sys import exit
from re import sub
from kmeans_vol import kmeans_vol
from glob import glob
from classifyReceptorSlices import classifyReceptorSlices
from receptorCrop import crop_source_files
from slicesToVolume import slices2vol
from utils.utils import downsample_y
from receptorInterpolate import receptorInterpolate, receptorSliceIndicator
from receptorAdjustAlignment import receptorRegister
from detectLines import apply_model

############################################
### Step 0: Train line detection network ###
############################################
#if train-model :
    #python3 detectLines.py --epochs 5 --step 0.2 --train-source "test/" --train-output "line_detection_model/"

clobber=False
mr_list=["MR1"]
hemisphere_list=["R"]
slab_list =["slab_1"] #, "slab_2", "slab_3", "slab_4", "slab_5", "slab_6"]
slabs=range(1,2)
#slab_list =["slab_1", "slab_2", "slab_3", "slab_4", ] #,
dont_remove_lines=["MR1/R_slab_6/"]
raw_source="raw/"
lin_source="lin/"

with open("scale_factors.json") as f : scale=json.load(f)

for mr in mr_list :
    #Setup Output Dir
    source_dir = mr

    for h in hemisphere_list :
        for i in slabs :
            slab=i
            dd = mr+os.sep+h+"_slab_"+str(slab)+os.sep
            source_lin_dir = 'lin'+os.sep+h+'_slab_'+str(i)
            source_raw_dir = raw_source+os.sep+h+"_slab_"+str(slab)

            lin_dwn_dir=dd+os.sep+'lin_dwn'+os.sep
            lines_output_dir = dd+"/lines_removed"
            crop_output_dir = dd+"/crop" 
            reg_output_dir = dd + os.sep + "coregistration"
            volume_dir =dd +os.sep+"/volume/"
            classify_dir =dd +os.sep+"/classify/"
            nonlinear_dir=dd+os.sep+"nonlinear/"

            if not os.path.exists(lin_dwn_dir) : 
                os.makedirs(lin_dwn_dir)
            
            ###################################
            ### Step 1 : Apply line removal ###
            ###################################
            #shell("python3 detectLines.py --step "+str(0.2)+"  --train-output line_detection_model/  --raw-source "+ raw_source+os.sep+h+"_slab_"+str(slab)+" --lin-source "+ lin_source+os.sep+h+"_slab_"+str(slab) +" --raw-output " + lines_output_dir+ " --ext .TIF") #--train-source \"test/\"
            #apply_model("line_detection_model/",source_raw_dir,source_lin_dir,lines_output_dir,0.2,clobber=False)

            ##################################
            ### Step 2: Automatic Cropping ###
            ##################################
            #crop_source_files(dd+"lines_removed/", crop_output_dir, downsample_step=0.2, manual_check=True,ext='.png')

            ####################################
            # Step 2.5 : Downsample Linearized #
            ####################################
            z_mm = scale[mr][h][str(i)]["size"]
            affine = np.array([[0.02,0,0,0],[0.0,0.02,0,0],[0,0,z_mm/4164.,0],[0,0,0,1]])
            #downsample_and_crop(source_lin_dir, lin_dwn_dir, crop_output_dir, affine)

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
                receptorRegister(lin_dwn_dir, reg_output_dir, tiers_str, rec_df_fn, ext=".png",  clobber=False)

            ####################################
            # Step 5. Classify receptor volume #
            ####################################
            print("Step 5. Classify receptor volume")
            out_file="vol_cls_"+str(i)+".nii.gz"
            classifyReceptorSlices(rec_fn, classify_dir, out_file, 5, False)
            
            ################################
            # Step 6. Resample cls and srv #
            ################################
            print("Step 6. Resample cls and srv")
            cls_fn = classify_dir + os.sep + "vol_cls_"+str(i)+".nii.gz"
            cls_iso_dwn_fn = classify_dir + os.sep + "vol_cls_"+str(i)+"_250um.nii.gz"
            if not os.path.exists(cls_iso_dwn_fn):
                downsample_y(cls_fn, cls_iso_dwn_fn, 0.25 )


            ###############################################
            ### Step 7: Transform MRI to receptor space ###
            ###############################################
            print("Step 7: Transform MRI to receptor space")
            srv = "srv/srv_space-rec-"+str(i)+"_lin.nii.gz"
            srv2cls_base=nonlinear_dir+os.sep+"transform_"+str(i)+"_"
            srv2cls=nonlinear_dir+os.sep+"warp_"+str(i)+"_Composite.h5"
            cls2srv=nonlinear_dir+os.sep+"warp_"+str(i)+"_InverseComposite.h5"
            srv_rsl=nonlinear_dir+os.sep+"srv_space-rec-"+str(i)+"_lin_rsl.nii.gz"
            output_file=nonlinear_dir+os.sep+"warped_"+str(i)+".nii"
            output_file_inverse=nonlinear_dir+os.sep+"warped_inverse_"+str(i)+".nii"

            if not os.path.exists(srv2cls) or not os.path.exists(cls2srv) : 
                if not os.path.exists(nonlinear_dir):
                    os.makedirs(nonlinear_dir)
                
                #ANTs Registration
                cmdline="antsRegistration --verbose 1 --float --collapse-output-transforms 1 --dimensionality 3 --initial-moving-transform [ "+cls_iso_dwn_fn+", "+srv+", 1 ] --initialize-transforms-per-stage 0 --interpolation Linear"
                cmdline+=" --transform Rigid[ 0.1 ] --metric Mattes[ "+cls_iso_dwn_fn+", "+srv+", 1, 32, Regular, 0.3 ] --convergence [ 1000x500x500, 1e-08, 10 ] --smoothing-sigmas 8.0x4.0x3.0vox --shrink-factors 8x4x3 --use-estimate-learning-rate-once 1 --use-histogram-matching 0 "
                cmdline += " --transform Affine[ 0.1 ] --metric Mattes[ "+cls_iso_dwn_fn+", "+srv+", 1, 32, Regular, 0.3 ] --convergence [ 1000x500x500 , 1e-08, 10 ] --smoothing-sigmas 8.0x4.0x2.0vox --shrink-factors 8x4x3 --use-estimate-learning-rate-once 1 --use-histogram-matching 0 "
                cmdline += " --transform SyN[ 0.1, 3.0, 0.0] --metric Mattes[ "+cls_iso_dwn_fn+", "+srv+", 0.5, 64, None ]  --convergence [ 2000x1500x10000x500, 1e-6,10 ] --smoothing-sigmas 8.0x6.0x4.0x2.0vox --shrink-factors 8x4x2x1  --winsorize-image-intensities [ 0.005, 0.995 ]  --write-composite-transform 1 "
                cmdline += "--output [ "+nonlinear_dir+"warp_"+str(slab)+"_ ,"+output_file+","+output_file_inverse+"]"
                #1500x1000x500x250
                print(cmdline)
                shell(cmdline)

            if not os.path.exists(srv_rsl) :
                cmdline=" ".join(["antsApplyTransforms -n Linear -d 3 -i",srv,"-r",cls_fn,"-t",srv2cls,"-o",srv_rsl])
                print(cmdline)
                shell(cmdline)            
            
            ############################################################
            ### 8. Interpolate missing slices for each receptor type ###
            ############################################################
            print("Step 8. Interpolate missing slices for each receptor type")
            ligands=["flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg,sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18,oxot,epib"]
            for ligand in ["flum"] :
                ligand_dir = dd + os.sep + "ligand" + os.sep + ligand + os.sep + 'slab-'+str(i)
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
                final_dir = dd + os.sep + "final"
                if not os.path.exists(dd+'/final') : os.makedirs(dd+'/final')
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

                receptorVolume_final_kmeans = final_dir + os.sep + final_base_fn+"_segmented.nii.gz"
                receptorVolume_final_dat = final_dir + os.sep + final_base_fn+"_segmented.dat"
                receptorVolume_final_kmeans_anlz = final_dir + os.sep + final_base_fn + "_segmented.img"
                if not os.path.exists( receptorVolume_final_kmeans ) or not os.path.exists(receptorVolume_final_dat) or clobber :
                    kmeans_vol(receptorVolumeRsl, 100,receptorVolume_final_dat, receptorVolume_final_kmeans)                
                shell("medcon -w -c anlz -f "+ receptorVolume_final_kmeans + ' -o '+receptorVolume_final_kmeans_anlz)

        exit(0)


