mkdir -p output output/srv

base_srv="output/mri1_gm_bg_srv.mnc"

itr=${1:-"0"}
clobber=${2:-0}
ITR="_itr-${itr}"

mkdir -p output/${ITR}

for i in `seq 1 1`; do
    ###############
    # Input files #
    ###############
    srv="output/srv/mri1_gm_srv_space-slab-${i}_trunc.mnc"
    if [[ "$itr" == "0"  ]]; then
        cls="../MR1/volume/vol_cls_${i}_morph_blur_bin.mnc.gz"
    else 
        let "itr0=itr-1"
        cls="output/_itr-${i0}/vol_cls_itr-${itr0}_${i}_morph_blur_bin.mnc.gz"
    fi
    ################
    # Output Files # 
    ################
    srv_rsl="output/${ITR}/vol_srv_itr-${itr}_${i}_rsl.mnc" #.gz"
    cls_rsl="output/${ITR}/vol_cls_itr-${itr}_${i}_morph_blur_bin_blur_rsl.mnc" #.gz"

    lin_tfm="output/${ITR}/srv_itr-${itr}_space-rec-${i}.tfm"
    lin_img="output/${ITR}/srv_itr-${itr}_space-rec-${i}_lin.mnc"
    cls_slab_space="output/${ITR}/vol_cls_itr-${itr}_${i}_morph_blur_bin_blur_rsl_space-slab-${i}.mnc"
    
    ########################
    # DOWNSAMPLE CLS & SRV #
    ########################
    ./downsample.sh $srv $srv_rsl 0.5 0.5 $clobber
    ./downsample.sh $cls $cls_rsl 0.5 0.5 $clobber
    
    ####################
    # LINEAR ALIGNMENT #
    ####################
    register $srv_rsl $cls_rsl
    if [[ ! -f $lin_tfm || "$clobber" -eq "1" ]] ; then
        echo Performing linear alignment of SRV to CLS
        ../bestlinreg_hires.sh $srv $cls $srv_rsl $cls_rsl $lin_tfm  
    fi
    clobber=1
    if [[ ! -f $lin_img || "$clobber" == 1 ]]; then
        echo Applying linear transformation
        mincresample -nearest -clobber -transformation $lin_tfm -like $cls_rsl $srv $lin_img
    fi
      
    if [[ ! -f $cls_slab_space || "$clobber" == "1" ]]; then
        echo Transforming CLS to SRV
        mincresample -clobber -transformation $lin_tfm -invert_transform -like $base_srv $cls_rsl $cls_slab_space
    fi
    clobber=1
done

concat_list=""
for i in `seq 1 6`; do
    mincresample -clobber -coronal output/srv/srv_space-rec-${i}_lin.mnc /tmp/tmp${i}.mnc
    minc_modify_header -dinsert yspace:step=-1 /tmp/tmp${i}.mnc
    concat_list="$concat_list /tmp/tmp${i}.mnc "
done
mincconcat -clobber -2 -nocheck_dimensions $concat_list output/srv/srv_space-rec_lin.mnc

exit 0

