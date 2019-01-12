base_srv="output/mri1_gm_bg_srv.mnc"

slab_fn="../MR1/coregistration/receptor_slices.csv"

for i in `seq 1 6`; do
    srv="output/srv/mri1_gm_srv_space-slab-${i}_trunc.mnc"
    mask="output/srv/mri1_brain_mask_space-mni_slab-${i}.mnc"
    if [[ ! -f ../MR1/volume/vol_cls_${i}.mnc ]]; then
            echo Error: could not find file ../MR1/volume/vol_cls_${i}.mnc
            exit 0
    fi
    ###############################################
    # Preprocess classified receptor volume (CLS) #
    ###############################################
    if [[ ! -f ../MR1/volume/vol_cls_${i}_morph_blur_bin.mnc.gz || "$clobber" == "1" ]]; then
        mincmorph -clobber -successive DDDEEEE -kernel 3x3x3_y.kern ../MR1/volume/vol_cls_${i}.mnc /tmp/morph.mnc
        mincblur -clobber -3dfwhm 0 1 0 -dimensions 2 /tmp/morph.mnc /tmp/morph
        mincmath -clobber  -gt -const 0.6 /tmp/morph_blur.mnc ../MR1/volume/vol_cls_${i}_morph_blur_bin.mnc
        gzip  ../MR1/volume/vol_cls_${i}_morph_blur_bin.mnc
    fi

    ########################################
    # Preprocess super-resolution GM image #
    ########################################
    #if [[ ! -f $mask.gz || "$clobber" == "1" ]]; then
    #    mincresample -nearest -clobber -like $base_srv  -transformation tfm_slab_${i}_to_mni.xfm  output/truncate/mr1_brain_mask_slab_${i}.mnc.gz $mask # /tmp/temp.mnc #
    #    gzip -f $mask
    #fi
    #mask=${mask}.gz
    if [[ ! -f  $srv  || "$clobber" == "1" ]]; then
        echo "Creating truncated version of super-resolution GM mask"
        #mincmath -quiet -clobber -mult $mask $base_srv $srv
        python3 slab_borders.py $slab_fn ../scale_factors.json $i $base_srv $srv 
        crop_from_mask $srv $srv $srv
        register $srv civet/mri1/final/mri1_t1_final.mnc
    fi

done
