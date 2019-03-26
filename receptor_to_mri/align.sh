#!/bin/bash

set -e

iterations=3 #${1:-0}
clobber=${2:-0}
out_base_dir="output-v4"
acc_all_fn="${out_base_dir}/acc.txt"
acc_all_png="${out_base_dir}/acc.png"
echo "Slab,Iteration,Test,Acc" > $acc_all_fn
clobber=0
for slab in `seq 3 3`; do
    printf "Slab: %d\n" "$slab"
    mkdir -p ${out_base_dir}/${slab}
    ##################
    # Base Variables #
    ##################
    scale_factors_fn="../scale_factors.json"
    base_srv="mri1_gm_bg_srv.mnc" #"input/mri1_gm_bg_srv.mnc"
    receptor_fn="../MR1/coregistration/receptor_slices.csv"
    cls0="../MR1/classify/vol_cls_${slab}.mnc.gz"
    cls00=$cls0

    ##################
    # SRV Preprocess #
    ##################
    # Outputs:
    srv="${out_base_dir}/srv_space-rec-${slab}_lin.mnc.gz"
    ./preprocess_srv.sh $base_srv $srv $slab $receptor_fn $scale_factors_fn 
    for itr in 0 ; do #`seq 0 $iterations`; do
        printf "\tIteration: %d\n" "$itr"
        out_dir=${out_base_dir}/${slab}/${itr}/
        mkdir -p ${out_dir}
        
        ##################
        # Preprocess CLS #
        ##################
        # Outputs :
        #cls="${out_dir}/vol_cls_itr-${itr}_${slab}_morph.mnc.gz"
        #./preprocess.sh $cls0 $cls $clobber
       	#exit 0 
        cls=$cls0
		#######################
        # 3D Align SRV to CLS #
        #######################
        cls_rsl="${out_dir}/vol_cls_itr-${itr}_${slab}_rsl.nii" #.gz"
		srv_nl="${out_dir}/result.1.mnc"
		./downsample.sh $cls $cls_rsl 0.25 0.25 $clobber
		echo $srv
		echo $cls_rsl

		if [[ ! -f ${srv%.*}.nii  || $clobber == 1 ]]; then
			mnc2nii  $srv ${srv%.*}.nii
		fi

		if [[ ! -f ${cls_rsl%.*}.nii  || $clobber == 1 ]]; then
			mnc2nii  $cls_rsl ${cls_rsl%.*}.nii
		fi

		if [[ ! -f $srv_nl || $clobber == 1 ]]; then
			elastix -f ${cls_rsl%.*}.nii -m ${srv%.*}.nii -p Parameters_Affine.txt -p Parameters_BSpline.txt -out ${out_dir}/
			nii2mnc ${out_dir}/result.1.nii $srv_nl 
		fi
        continue  
		##################################
        # Refine 2D slice-wise alignment #
        ##################################
        # Outputs :
        cls_2d_rsl="${out_dir}/vol_cls_itr-${itr}_${slab}_2D-rsl.nii.gz"
        valid_slices="${out_base_dir}/${slab}/valid_slices_${slab}.txt"
        #./2d_alignment.sh $cls0 $cls $srv_nl $valid_slices $out_dir $cls_2d_rsl $clobber
        
        ###############################################################
        # Calculate Dice of cls_2d_rsl compared to previous iteration #
        ###############################################################
        cls0=$cls_2d_rsl
    done
done

python plot_acc.py $acc_all_fn $acc_all_png
# Ligand
#./ligand.sh

