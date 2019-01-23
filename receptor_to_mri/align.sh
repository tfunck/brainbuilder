iterations=1 #${1:-0}
clobber=${2:-0}
out_base_dir="output-v1"
acc_all_fn="${out_base_dir}/acc.txt"
acc_all_png="${out_base_dir}/acc.png"
echo "Slab,Iteration,Test,Acc" > $acc_all_fn

for slab in `seq 1 1`; do
    printf "Slab: %d\n" "$slab"
    mkdir -p ${out_base_dir}/${slab}
    ##################
    # Base Variables #
    ##################
    scale_factors_fn="../scale_factors.json"
    base_srv="mri1_gm_bg_srv.mnc" #"input/mri1_gm_bg_srv.mnc"
    receptor_fn="../MR1/coregistration/receptor_slices.csv"
    cls0="../MR1/volume/vol_cls_${slab}.mnc.gz"
    cls00=$cls0

    ##################
    # SRV Preprocess #
    ##################
    # Outputs:
    srv="${out_base_dir}/srv_space-rec-${slab}_lin.mnc.gz"
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
        ./preprocess.sh $cls0 $cls $clobber
        
        #######################
        # 3D Align SRV to CLS #
        #######################
        # Outputs :
        srv_rsl="${out_dir}/vol_srv_itr-${itr}_${slab}_rsl.mnc" #.gz"
        cls_rsl="${out_dir}/vol_cls_itr-${itr}_${slab}_rsl.mnc" #.gz"
        srv_lin_rsl="${out_dir}/vol_srv_itr-${itr}_${slab}_lin_rsl.mnc" #.gz"
        tfm_lin="${out_dir}/srv_itr-${itr}_space-rec-${slab}_lin.tfm"
        tfm_nl="${out_dir}/srv_itr-${itr}_space-rec-${slab}_nl.tfm"
        srv_lin="${out_dir}/srv_itr-${itr}_space-rec-${slab}_lin.mnc"
        srv_nl="${out_dir}/srv_itr-${itr}_space-rec-${slab}_nl.mnc"
        ./3d_alignment.sh $itr $cls $srv $srv_rsl $cls_rsl $srv_lin_rsl $tfm_lin $tfm_nl $srv_lin $srv_nl $clobber
        
        ##################################
        # Refine 2D slice-wise alignment #
        ##################################
        # Outputs :
        cls_2d_rsl="${out_dir}/vol_cls_itr-${itr}_${slab}_morph_blur_bin_2D-rsl.mnc"
        valid_slices="${out_base_dir}/${slab}/valid_slices_${slab}.txt"
        ./2d_alignment.sh $cls0 $cls $srv_nl $valid_slices $out_dir $cls_2d_rsl $clobber
        
        ###############################################################
        # Calculate Dice of cls_2d_rsl compared to previous iteration #
        ###############################################################
        # Outputs 
        acc_2d_accuracy_fn=${out_dir}/acc_slab-${slab}_itr-${itr}_accuracy_2d.txt
        acc_3d_accuracy_fn=${out_dir}/acc_slab-${slab}_itr-${itr}_accuracy_3d.txt
        clobber=1
        ./calc_acc.sh $srv_lin $cls0 $slab $itr "3D-lin" $acc_3d_accuracy_fn $acc_all_fn $clobber 
        if [[ "$srv_nl" != "$srv_lin" ]] ; then
            ./calc_acc.sh $srv_nl $cls0 $slab $itr "3D-nl" $acc_3d_accuracy_fn $acc_all_fn $clobber 
        fi
        ./calc_acc.sh $srv_nl $cls_2d_rsl $slab $itr "2D" $acc_2d_accuracy_fn $acc_all_fn $clobber 
        clobber=0
        cls0=$cls_2d_rsl
    done
done

python plot_acc.py $acc_all_fn $acc_all_png
# Ligand
#./ligand.sh

