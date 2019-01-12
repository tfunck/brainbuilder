       nl_tfm="output/${ITR}/srv_itr-${itr0}_space-rec-${i}_nl.tfm"
    nl_img="output/${ITR}/srv_itr-${itr0}_space-rec-${i}_nl.mnc"
    final_tfm="output/${ITR}/srv_itr-${itr0}_space-rec-${i}_lin+nl.tfm"
    ########################
    # NON-LINEAR ALIGNMENT #
    ########################
    if [[ ! -f $nl_tfm || "$clobber" == 1 ]]; then
        echo Performing linear transformation
        ../bestnlreg_hires.sh $lin_img $cls $srv_rsl  $cls_rsl  output/srv/srv_space-rec-${i}.tfm $nl_tfm output/nl
    fi

    if [[ ! -f $final_tfm || "$clobber" == 1 ]]; then
        echo Concatenating Linear and Nonlinear Transforms
        xfmconcat -clobber output/srv/srv_space-rec-${i}.tfm $nl_tfm $final_tfm 
    fi
    
    if [[ ! -f $nl_img || "$clobber" == 1 ]]; then
        echo Applying nonlinear transformation
        mincresample -tricubic -clobber -transformation $final_tfm -like $cls_rsl $srv_rsl $nl_img
    fi
#for i in `seq 1 6`; do
#    cp output/srv/srv_space-rec-${i}_nl.mnc /tmp/tmp${i}.mnc
#    minc_modify_header -dinsert yspace:step=-1 /tmp/tmp${i}.mnc
#    concat_list="$concat_list /tmp/tmp${i}.mnc "
#done
#mincconcat -clobber -2 -nocheck_dimensions $concat_list output/srv/srv_space-rec_nl.mnc

#concat_list=""
#for i in `seq 1 6`; do
#    concat_list="$concat_list output/srv/vol_cls_${i}_morph_blur_bin_blur_rsl_space-slab-${i}.mnc "
#done
#minccalc -clobber -expr "sum(A)" $concat_list output/srv/vol_cls_nl.mnc
