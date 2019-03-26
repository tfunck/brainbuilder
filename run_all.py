import os
from utils.utils import shell
from sys import exit
from re import sub
from receptorRegister import receptorRegister
from glob import glob
from classifyReceptorSlices import classifyReceptorSlices
from slices2vol import slices2vol
from utils.utils import downsample_y
from receptorInterpolate import receptorInterpolate


############################################
### Step 0: Train line detection network ###
############################################
#if train-model :
    #python3 detectLines.py --epochs 5 --step 0.2 --train-source "test/" --train-output "line_detection_model/"

mr_list=["MR1"]
hemisphere_list=["R"]
slab_list =["slab_1"] #, "slab_2", "slab_3", "slab_4", "slab_5", "slab_6"]
slabs=range(1,2)
#slab_list =["slab_1", "slab_2", "slab_3", "slab_4", ] #,
dont_remove_lines=["MR1/R_slab_6/"]
raw_source="raw/"
lin_source="lin/"
for mr in mr_list :
    for h in hemisphere_list :
        for slab in slab_list :
            dd = mr+os.sep+h+"_"+slab+os.sep
            ###################################
            ### Step 1 : Apply line removal ###
            ###################################
            #shell("python3 detectLines.py --step "+str(0.2)+" --train-source \"test/\" --train-output line_detection_model/  --raw-source "+ raw_source+os.sep+h+"_"+slab+" --raw-output "+dd +"lines_removed")

            ##################################
            ### Step 2: Automatic Cropping ###
            ##################################
            noframe=""
            #if dd in dont_remove_lines :
            #    noframe="--no-frame"
            #shell("python3 receptorCrop.py  "+noframe+" --source "+dd+"lines_removed/ --output "+dd+"/crop --step "+str(0.2)+" --ext \".png\" ")
        
        ##################################
        ### Step 3 : Receptor Register ###
        ##################################
        print(" Step 3 : Receptor Register")
        source_dir = mr
        reg_output_dir = mr + os.sep + "coregistration_ants"
        slice_order_fn= "section_numbers.csv"
        tiers_str = "flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg;sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18;oxot;epib"
        clobber=False
        #receptorRegister(source_dir=source_dir, output_dir=reg_output_dir, slice_order_fn=slice_order_fn,slabs_to_run=slabs, clobber=clobber, tiers_str=tiers_str )
       
        
        ############################################
        ### Step 4: Reconstruct slices to volume ###
        ############################################
        print("Step 4: Reconstruct slices to volume")
        receptor_fn=mr+os.sep+"coregistration/receptor_slices.csv"
        source_dir =mr+os.sep+"/coregistration_ants/resample/"
        output_dir =mr +os.sep+"/volume_ants/"
        
        slices2vol(receptor_fn, source_dir, output_dir, slabs)
        
        for i in slabs :
            ####################################
            # Step 5. Classify receptor volume #
            ####################################
            print("Step 5. Classify receptor volume")
            in_file =mr +os.sep+"/volume_ants/vol_"+str(i)+".nii.gz"
            out_dir =mr +os.sep+"/classify_ants/"
            out_file="vol_cls_"+str(i)+".nii.gz"
            classifyReceptorSlices(in_file, out_dir,out_file,5,False)
        
            ################################
            # Step 6. Resample cls and srv #
            ################################
            print("Step 6. Resample cls and srv")
            fixed_image = mr + os.sep + "/classify_ants" + os.sep + "vol_cls_"+str(i)+".nii.gz"
            fixed_image_rsl = mr + os.sep + "/classify_ants" + os.sep + "vol_cls_"+str(i)+"_250um.nii.gz"
            if not os.path.exists(fixed_image_rsl):
                downsample_y(fixed_image, fixed_image_rsl, 0.25 )
        
            ###############################################
            ### Step 7: Transform MRI to receptor space ###
            ###############################################
            print("Step 7: Transform MRI to receptor space")
            fixed_image = mr + os.sep + "/classify_ants" + os.sep + "vol_cls_"+str(i)+"_250um.nii.gz"
            volume = mr + os.sep + "/classify_ants" + os.sep + "vol_cls_"+str(i)+".nii.gz"
            moving_image = "srv/srv_space-rec-"+str(i)+"_lin.nii.gz"
            output_dir=mr+os.sep+"nonlinear_ants/"
            srv2cls_base=output_dir+os.sep+"transform_"+str(i)+"_"
            srv2cls=output_dir+os.sep+"warp_slab_"+str(i)+"_Composite.h5"
            cls2srv=output_dir+os.sep+"warp_slab_"+str(i)+"_InverseComposite.h5"
            moving_rsl=output_dir+os.sep+"srv_space-rec-"+str(i)+"_lin_rsl.nii.gz"
            output_file=output_dir+os.sep+"warped_"+str(i)+".nii"
            output_file_inverse=output_dir+os.sep+"warped_inverse_"+str(i)+".nii"

            if not os.path.exists(srv2cls) or not os.path.exists(cls2srv) : 
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                #ANTs Registration
                cmdline="antsRegistration --verbose 1 --float --collapse-output-transforms 1 --dimensionality 3 --initial-moving-transform [ "+fixed_image+", "+moving_image+", 1 ] --initialize-transforms-per-stage 0 --interpolation Linear"
                cmdline+=" --transform Rigid[ 0.1 ] --metric Mattes[ "+fixed_image+", "+moving_image+", 1, 32, Regular, 0.3 ] --convergence [ 1000x500x500, 1e-08, 10 ] --smoothing-sigmas 8.0x4.0x3.0vox --shrink-factors 8x4x3 --use-estimate-learning-rate-once 1 --use-histogram-matching 0 "
                cmdline += " --transform Affine[ 0.1 ] --metric Mattes[ "+fixed_image+", "+moving_image+", 1, 32, Regular, 0.3 ] --convergence [ 1000x500x500 , 1e-08, 10 ] --smoothing-sigmas 8.0x4.0x2.0vox --shrink-factors 8x4x3 --use-estimate-learning-rate-once 1 --use-histogram-matching 0 "
                cmdline += " --transform SyN[ 0.1, 3.0, 0.0] --metric Mattes[ "+fixed_image+", "+moving_image+", 0.5, 64, None ]  --convergence [ 2000x1500x10000x500, 1e-6,10 ] --smoothing-sigmas 8.0x6.0x4.0x2.0vox --shrink-factors 8x4x2x1  --winsorize-image-intensities [ 0.005, 0.995 ]  --write-composite-transform 1 "
                cmdline += " --output [ "+output_dir+"warp_"+str(slab)+"_ , "+output_file+", "+output_file_inverse+" ] "
                #1500x1000x500x250
                print(cmdline)
                #shell(cmdline)

            if not os.path.exists(moving_rsl) :

                cmdline=" ".join(["antsApplyTransforms -n Linear -d 3 -i",moving_image,"-r",volume,"-t ",srv2cls,"-o",moving_rsl])
                print(cmdline)
                shell(cmdline)            

            ############################################################
            ### 8. Interpolate missing slices for each receptor type ###
            ############################################################
            print("Step 8. Interpolate missing slices for each receptor type")
            slice_info_fn=mr+os.sep+"coregistration_ants/receptor_slices.csv"
            ligands=["flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg,sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18,oxot,epib"]
            for ligand in ["flum"] :
                print("\tLigand:",ligand)
                output_dir = mr + os.sep + "ligand" + os.sep + ligand + os.sep
                receptorVolume = mr + os.sep + "ligand" + os.sep + ligand + os.sep + ligand + ".nii.gz"
                cls_fn= mr+os.sep+"/classify"+os.sep+"vol_cls_"+str(i)+".nii.gz"        
                srv_fn= mr+os.sep+"nonlinear_ants/srv_space-rec-"+str(i)+"_lin_rsl.nii.gz"
                rec_fn=mr+os.sep+"/volume"+os.sep+"vol_"+str(i)+".nii.gz"
                receptorInterpolate( i, rec_fn, srv_fn, cls_fn, output_dir, ligand, slice_info_fn)
        
                ######################################################
                ### 9. Transform Linearized tif files to MRI space ###
                ######################################################
                receptorVolume_final = mr + os.sep + "final" + os.sep + ligand + "_space-srv.nii.gz"
                cmdline=" ".join(["antsApplyTransforms -e 1 -d 3 -n Linear -i",receptorVolume,"-r",cls_fn,"-t",srv2cls,"-o",receptorVolume_final])
                print(cmdline)
                shell(cmdline)

        exit(0)


