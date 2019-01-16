iterations=4 #${1:-0}
clobber=${2:-0}
out_base_dir="output-v2"
acc_all_fn="${out_base_dir}/acc.txt"
acc_all_png="${out_base_dir}/acc.png"
echo "" > /tmp/log.txt
echo "Slab,Iteration,Test,Acc" > $acc_all_fn

for slab in `seq 1 1`; do
    printf "Slab: %d\n" "$slab"
    mkdir -p ${out_base_dir}/${slab}
    ##################
    # Base Variables #
    ##################
    scale_factors_fn=`./check_file.sh "../scale_factors.json"`
    base_srv=`./check_file.sh "input/mri1_gm_bg_srv.mnc"`
    receptor_fn=`./check_file.sh "../MR1/coregistration/receptor_slices.csv"`
    cls0=`./check_file.sh "../MR1/volume/vol_cls_${slab}.mnc.gz"`
    cls00=$cls0
    clobber=0 
    ##################
    # SRV Preprocess #
    ##################
    # Outputs:
    srv="${out_base_dir}/srv_space-rec-${slab}_lin.mnc"
    ./preprocess_srv.sh $base_srv $srv $slab $receptor_fn $scale_factors_fn 
    for itr in `seq 0 $iterations`; do
        printf "\tIteration: %d\n" "$itr"
        out_dir=${out_base_dir}/${slab}/${itr}/
        mkdir -p ${out_dir}
        
        ##################
        # Preprocess CLS #
        ##################
        # Outputs :
        cls="${out_dir}/vol_cls_itr-${itr}_${slab}_morph_blur_bin.mnc.gz"
        clobber=0
        ./preprocess.sh $cls0 $cls $clobber
        #if [[ $itr == 1 ]]; then
        #    ls $cls
        #    ls output-v2/1/0//vol_cls_itr-0_1_morph_blur_bin.mnc.gz
        #    register ${out_base_dir}/${slab}/0/vol_cls_itr-0_${slab}_morph_blur_bin.mnc.gz output-v2/1/0//vol_cls_itr-0_1_morph_blur_bin.mnc.gz
        #fi
        
        #######################
        # 3D Align SRV to CLS #
        #######################
        # Outputs :
        srv_rsl="${out_dir}/vol_srv_itr-${itr}_${slab}_rsl.mnc" #.gz"
        cls_rsl="${out_dir}/vol_cls_itr-${itr}_${slab}_rsl.mnc" #.gz"
        tfm_lin="${out_dir}/srv_itr-${itr}_space-rec-${slab}.tfm"
        srv_lin="${out_dir}/srv_itr-${itr}_space-rec-${slab}_lin.mnc"

        ./3d_alignment.sh $cls $srv $srv_rsl $cls_rsl $tfm_lin $srv_lin

        ##################################
        # Refine 2D slice-wise alignment #
        ##################################
        # Outputs :
        cls_2d_rsl="${out_dir}/vol_cls_itr-${itr}_${slab}_morph_blur_bin_2D-rsl.mnc"
        valid_slices="${out_base_dir}/${slab}/valid_slices_${slab}.txt"
        ./2d_alignment.sh $cls0 $cls $srv_lin $valid_slices $out_dir $cls_2d_rsl $clobber
        
        ###############################################################
        # Calculate Dice of cls_2d_rsl compared to previous iteration #
        ###############################################################
        # Outputs 
        acc_change_fn=${out_dir}/acc_slab-${slab}_itr-${itr}_change.txt
        acc_accuracy_fn=${out_dir}/acc_slab-${slab}_itr-${itr}_accuracy.txt
        clobber=1
        ./calc_acc.sh $cls00 $cls_2d_rsl $slab $itr "Change"  $acc_change_fn $acc_all_fn $clobber 
        ./calc_acc.sh $srv_lin $cls_2d_rsl $slab $itr "Accuracy" $acc_accuracy_fn $acc_all_fn $clobber 
        clobber=0
        cls0=$cls_2d_rsl
    done
done

python plot_acc.py $acc_all_fn $acc_all_png
# Ligand
#./ligand.sh

