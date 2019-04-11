import os
import json
import numpy as np
from utils.utils import shell
from sys import exit
from re import sub
from receptorRegister import receptorRegister
from glob import glob
from classifyReceptorSlices import classifyReceptorSlices
from slicesToVolume import slices2vol
from utils.utils import downsample_y
from receptorInterpolate import receptorInterpolate
from integrate_downsample import integrate_downsample_tif

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
    nonlinear_dir=mr+os.sep+"nonlinear_ants/"
    source_dir = mr
    reg_output_dir = mr + os.sep + "coregistration_ants"
    source_dir =mr+os.sep+"/coregistration_ants/resample/"
    volume_dir =mr +os.sep+"/volume_ants/"
    classify_dir =mr +os.sep+"/classify_ants/"

    for h in hemisphere_list :
        for i in slabs :
            slab=i
            dd = mr+os.sep+h+"_slab_"+str(slab)+os.sep
            source_lin_dir = 'lin'+os.sep+h+'_slab_'+str(i)
            lin_dir=mr+os.sep+'lin'+os.sep+h+'_slab_'+str(i) + os.sep

            if not os.path.exists(lin_dir) : 
                os.makedirs(lin_dir)

            ##################################
            # Step 0 : Downsample Linearized #
            ##################################
            z_mm = scale[mr][h][str(i)]["size"]
            affine = np.array([[0.02,0,0,0],[0.0,0.02,0,0],[0,0,z_mm/4164.,0],[0,0,0,1]])
            for f in glob(source_lin_dir+"/*.TIF") :
                dwn_fn = lin_dir + os.path.basename(f)
                if not os.path.exists(dwn_fn) or clobber :
                    integrate_downsample_tif(f, 0.2, affine, dwn_fn)

            ###################################
            ### Step 1 : Apply line removal ###
            ###################################
            shell("python3 detectLines.py --step "+str(0.2)+"  --train-output line_detection_model/  --raw-source "+ raw_source+os.sep+h+"_slab_"+str(slab)+" --lin-source "+ lin_source+os.sep+h+"_slab_"+str(slab) +" --raw-output "+dd +"lines_removed_lin --ext .TIF") #--train-source \"test/\"
            exit(0)
            ##################################
            ### Step 2: Automatic Cropping ###
            ##################################
            noframe=""
            #if dd in dont_remove_lines :
            #    noframe="--no-frame"
            shell("python3 receptorCrop.py  "+noframe+" --source "+dd+"lines_removed/ --output "+dd+"/crop --step "+str(0.2)+" --ext \".png\" ")
        
            ##################################
            ### Step 3 : Receptor Register ###
            ##################################
            print(" Step 3 : Receptor Register")

            slice_order_fn= "section_numbers.csv"
            tiers_str = "flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg;sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18;oxot;epib"
            slabs_to_run=i
            clobber=False
            #receptorRegister(source_dir=source_dir, output_dir=reg_output_dir, slice_order_fn=slice_order_fn,slabs_to_run=slabs, clobber=clobber, tiers_str=tiers_str )
           
            ############################################
            ### Step 4: Reconstruct slices to volume ###
            ############################################
            print("Step 4: Reconstruct slices to volume")
            receptor_fn=mr+os.sep+"coregistration/receptor_slices.csv"

            slices2vol(receptor_fn, source_dir, volume_dir, [i])
            slices2vol(receptor_fn, lin_rsl_dir, volume_dir+"/lin", [i], ext=".TIF")
            exit(0) 
            ####################################
            # Step 5. Classify receptor volume #
            ####################################
            print("Step 5. Classify receptor volume")
            in_file =mr +os.sep+"/volume_ants/vol_"+str(i)+".nii.gz"
            out_file="vol_cls_"+str(i)+".nii.gz"
            classifyReceptorSlices(in_file, classify_dir,out_file,5,False)
        
            ################################
            # Step 6. Resample cls and srv #
            ################################
            print("Step 6. Resample cls and srv")
            cls = mr + os.sep + "/classify_ants" + os.sep + "vol_cls_"+str(i)+".nii.gz"
            cls_iso_dwn = mr + os.sep + "/classify_ants" + os.sep + "vol_cls_"+str(i)+"_250um.nii.gz"
            if not os.path.exists(cls_iso_dwn):
                downsample_y(cls, cls_iso_dwn, 0.25 )
        
            ###############################################
            ### Step 7: Transform MRI to receptor space ###
            ###############################################
            print("Step 7: Transform MRI to receptor space")
            srv = "srv/srv_space-rec-"+str(i)+"_lin.nii.gz"
            srv2cls_base=nonlinear_dir+os.sep+"transform_"+str(i)+"_"
            srv2cls=nonlinear_dir+os.sep+"warp_slab_"+str(i)+"_Composite.h5"
            cls2srv=nonlinear_dir+os.sep+"warp_slab_"+str(i)+"_InverseComposite.h5"
            srv_rsl=nonlinear_dir+os.sep+"srv_space-rec-"+str(i)+"_lin_rsl.nii.gz"
            output_file=nonlinear_dir+os.sep+"warped_"+str(i)+".nii"
            output_file_inverse=nonlinear_dir+os.sep+"warped_inverse_"+str(i)+".nii"

            if not os.path.exists(srv2cls) or not os.path.exists(cls2srv) : 
                if not os.path.exists(nonlinear_dir):
                    os.makedirs(nonlinear_dir)
                
                #ANTs Registration
                cmdline="antsRegistration --verbose 1 --float --collapse-output-transforms 1 --dimensionality 3 --initial-moving-transform [ "+cls_iso_dwn+", "+srv+", 1 ] --initialize-transforms-per-stage 0 --interpolation Linear"
                cmdline+=" --transform Rigid[ 0.1 ] --metric Mattes[ "+cls_iso_dwn+", "+srv+", 1, 32, Regular, 0.3 ] --convergence [ 1000x500x500, 1e-08, 10 ] --smoothing-sigmas 8.0x4.0x3.0vox --shrink-factors 8x4x3 --use-estimate-learning-rate-once 1 --use-histogram-matching 0 "
                cmdline += " --transform Affine[ 0.1 ] --metric Mattes[ "+cls_iso_dwn+", "+srv+", 1, 32, Regular, 0.3 ] --convergence [ 1000x500x500 , 1e-08, 10 ] --smoothing-sigmas 8.0x4.0x2.0vox --shrink-factors 8x4x3 --use-estimate-learning-rate-once 1 --use-histogram-matching 0 "
                cmdline += " --transform SyN[ 0.1, 3.0, 0.0] --metric Mattes[ "+cls_iso_dwn+", "+srv+", 0.5, 64, None ]  --convergence [ 2000x1500x10000x500, 1e-6,10 ] --smoothing-sigmas 8.0x6.0x4.0x2.0vox --shrink-factors 8x4x2x1  --winsorize-image-intensities [ 0.005, 0.995 ]  --write-composite-transform 1 "
                cmdline += "--output [ "+nonlinear_dir+"warp_"+str(slab)+"_ ,"+output_file+","+output_file_inverse+"]"
                #1500x1000x500x250
                print(cmdline)
                #shell(cmdline)

            if not os.path.exists(srv_rsl) :
                cmdline=" ".join(["antsApplyTransforms -n Linear -d 3 -i",srv,"-r",cls,"-t",srv2cls,"-o",srv_rsl])
                print(cmdline)
                shell(cmdline)            

            ############################################################
            ### 8. Interpolate missing slices for each receptor type ###
            ############################################################
            print("Step 8. Interpolate missing slices for each receptor type")
            slice_info_fn=mr+os.sep+"coregistration_ants/receptor_slices.csv"
            ligands=["flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg,sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18,oxot,epib"]
            for ligand in ["flum"] :
                ligand_dir = mr + os.sep + "ligand" + os.sep + ligand + os.sep + 'slab-'+str(i)
                print("\tLigand:",ligand)
                receptorVolume = ligand_dir + os.sep + ligand + ".nii.gz"
                receptorVolumeRsl = ligand_dir + os.sep + ligand + "_250um.nii.gz"
                cls_fn= mr+os.sep+"/classify"+os.sep+"vol_cls_"+str(i)+".nii.gz"        
                rec_fn=mr+os.sep+"/volume"+os.sep+"vol_"+str(i)+".nii.gz"
                receptorInterpolate( i, rec_fn, srv_rsl, cls_fn, ligand_dir, ligand, slice_info_fn, reg_output_dir+'/transforms', lin_dir)
                exit(0)
                #downsample_y(receptorVolume, receptorVolumeRsl, 1.25 )
                
                ######################################################
                ### 9. Transform Linearized tif files to MRI space ###
                ######################################################
                print("9. Transform Linearized tif files to MRI space")
                if not os.path.exists('MR1/final') : os.makedirs('MR1/final')
                receptorVolume_final = mr + os.sep + "final" + os.sep + ligand + "_space-srv.nii.gz"
                cmdline=" ".join(["antsApplyTransforms -d 3 -n Linear  -i",receptorVolumeRsl,"-r",srv,"-t",cls2srv,"-o",receptorVolume_final])
                #cmdline=" ".join(["antsApplyTransforms -d 3 -n Linear  -i", cls_iso_dwn,"-r",srv,"-t",cls2srv,"-o",receptorVolume_final])
                print(cmdline)
                shell(cmdline)

        exit(0)


