mkdir -p output output/srv

downsample() {
        vol=$1
        s=$2
        fwhm_1=$3
        clobber_1=$4
        base=`echo $vol | sed 's/.gz//' | sed 's/.mnc//'`
        blur=`echo $vol | sed 's/.gz//' | sed 's/.mnc/_blur.mnc/'`
        blur_rsl=`echo $vol | sed 's/.gz//'| sed 's/.mnc/_blur_rsl.mnc/'`

        echo $base
        echo $blur 
        echo $clobber_1
        if [[ ! -f $blur || $clobber_1 == 1 ]]; then
            echo
            echo Smoothing $base.mnc.gz with $fwhm_1 FWHM 
            echo
            echo mincblur -clobber -fwhm $fwhm_1 $vol $base
            mincblur -clobber -fwhm $fwhm_1 $vol $base
             
        fi
        if [[ ! -f $blur_rsl || $clobber_1 == 1 ]]; then
            echo
            echo Resampling $blur to step size $s
            echo
            z=`mincinfo -dimlength zspace $vol`
            y=`mincinfo -dimlength yspace $vol`
            x=`mincinfo -dimlength xspace $vol`
            zstep=`mincinfo -attvalue zspace:step $vol`
            ystep=`mincinfo -attvalue yspace:step $vol`
            xstep=`mincinfo -attvalue xspace:step $vol`
            zstart=`mincinfo -attvalue zspace:start $vol`
            ystart=`mincinfo -attvalue yspace:start $vol`
            xstart=`mincinfo -attvalue xspace:start $vol`

            zw=`echo "(($z * $zstep)  ) " | bc`
            yw=`echo "(($y * $ystep)  ) " | bc`
            xw=`echo "(($x * $xstep)  ) " | bc`

            zmax=`echo "$zw / $s " | bc`
            ymax=`echo "$yw / $s " | bc`
            xmax=`echo "$xw / $s " | bc`
            step="$s $s $s"
            echo Start: $zstart $ystart $xstart
            echo Step:  $zstep $ystep $xstep
            echo World Max: $zw $yw $xw
            echo Voxel Max:   $z $y $x
            echo New Voxel Max:  $zmax $ymax $xmax
            echo mincresample -clobber -step $step -nelements $xmax $ymax $zmax  $blur $blur_rsl
            mincresample -clobber -step $step -nelements $xmax $ymax $zmax  $blur $blur_rsl
        fi
        return 0
}

#-72,-126,-90
base_srv="output/mri1_gm_bg_srv.mnc"
for i in `seq 1 6`; do
    if [[ ! -f ../MR1/volume/vol_cls_${i}.mnc ]]; then
            echo Error: could not find file ../MR1/volume/vol_cls_${i}.mnc
            exit 0
    fi

    ###############
    # Input files #
    ###############
    srv="output/srv/mri1_gm_srv_space-slab-${i}_trunc.mnc"
    cls="../MR1/volume/vol_cls_${i}_morph_blur_bin.mnc"
    rec="../MR1/volume/vol_${i}.mnc.gz"

    ################
    # Output Files # 
    ################
    srv_rsl=output/srv/mri1_gm_srv_space-slab-${i}_trunc_blur_rsl.mnc
    cls_rsl=../MR1/volume/vol_cls_${i}_morph_blur_bin_blur_rsl.mnc

    
    ###############################################
    # Preprocess classified receptor volume (CLS) #
    ###############################################
    if [[ ! -f ../MR1/volume/vol_cls_${i}_morph_blur_bin.mnc.gz || "$clobber" == "1" ]]; then
        mincmorph -clobber -successive DDDEEEE -kernel 3x3x3_y.kern ../MR1/volume/vol_cls_${i}.mnc /tmp/morph.mnc
        mincblur -clobber -3dfwhm 0 1 0 -dimensions 2 /tmp/morph.mnc /tmp/morph
        mincmath -clobber  -gt -const 0.6 /tmp/morph_blur.mnc ../MR1/volume/vol_cls_${i}_morph_blur_bin.mnc
        gzip  ../MR-1/volume/vol_cls_${i}_morph_blur_bin.mnc
    fi
    clobber=0
    ########################################
    # Preprocess super-resolution GM image #
    ########################################
    mask="output/srv/mri1_brain_mask_space-mni_slab-${i}.mnc"
    if [[ ! -f $mask.gz || "$clobber" == "1" ]]; then
    #-step 0.25 0.25 0.25 
        mincresample -nearest -clobber -like $base_srv  -transformation tfm_slab_${i}_to_mni.xfm  output/truncate/mr1_brain_mask_slab_${i}.mnc.gz $mask # /tmp/temp.mnc #
        #mincresample -nearest -clobber /tmp/temp.mnc -like  output/srv/mri1_gm_srv_space-slab-${i}.mnc $mask
        gzip -f $mask
    fi
    mask=${mask}.gz
    
    if [[ ! -f  $srv  || "$clobber" == "1" ]]; then
        echo "Creating truncated version of super-resolution GM mask"
        mincmath -quiet -clobber -mult $mask $base_srv $srv
        crop_from_mask $srv $srv $srv
    fi
   
    ########################
    # DOWNSAMPLE CLS & SRV #
    ########################
    downsample $srv 1 0.5 $clobber
    downsample $cls 1 0.5 $clobber
    echo $srv_rsl
    echo $cls_rsl

    ####################
    # LINEAR ALIGNMENT #
    ####################
    lin_tfm=output/srv/srv_space-rec-${i}.tfm
    lin_img=output/srv/srv_space-rec-${i}_lin.mnc
    if [[ ! -f $lin_tfm || "$clobber" -eq "1" ]] ; then
        echo Performing linear alignment of SRV to CLS
        ../bestlinreg_hires.sh $srv $cls $srv_rsl $cls_rsl $lin_tfm  
    fi
    
    if [[ ! -f $lin_img || "$clobber" == 1 ]]; then
        echo Applying linear transformation
        mincresample -nearest -clobber -transformation $lin_tfm -like $cls_rsl $srv $lin_img
    fi

    ########################
    # NON-LINEAR ALIGNMENT #
    ########################
    nl_tfm="output/nl/srv_space-rec-${i}_nl.tfm"
    nl_img="output/srv/srv_space-rec-${i}_nl.mnc"
    if [[ ! -f $nl_tfm || "$clobber" == 1 ]]; then
        echo Performing linear transformation
        ../bestnlreg_hires.sh $lin_img $cls $srv_rsl  $cls_rsl  output/srv/srv_space-rec-${i}.tfm $nl_tfm output/nl
    fi

    final_tfm=output/srv/srv_space-rec-${i}_lin+nl.tfm
    if [[ ! -f $final_tfm || "$clobber" == 1 ]]; then
        echo Concatenating Linear and Nonlinear Transforms
        xfmconcat -clobber output/srv/srv_space-rec-${i}.tfm $nl_tfm $final_tfm 
    fi
    
    if [[ ! -f $nl_img || "$clobber" == 1 ]]; then
        echo Applying nonlinear transformation
        mincresample -tricubic -clobber -transformation $final_tfm -like $cls_rsl $srv_rsl $nl_img
    fi

    cls_slab_space=output/srv/vol_cls_${i}_morph_blur_bin_blur_rsl_space-slab-${i}.mnc
    if [[ ! -f $cls_slab_space || "$clobber" == "1" ]]; then
        echo Transforming CLS to SRV
        mincresample -clobber -transformation $final_tfm -invert_transform -like $base_srv  $cls_rsl $cls_slab_space
    fi

    #rec_nl_rsl=output/srv/vol_${i}_in_blur_rsl_space-slab-${i}.mnc
    #if [[ $rec_nl_rsl  ]]; then
    #    mincresample -tricubic -clobber -transformation $final_tfm  -tfm_input_sampling $rec $rec_nl_rsl
    #fi
    #register $cls_rsl output/srv/srv_space-rec-${i}_nl.mnc
done

for i in `seq 1 6`; do
    cp output/srv/srv_space-rec-${i}_nl.mnc /tmp/tmp${i}.mnc
    minc_modify_header -dinsert yspace:step=-1 /tmp/tmp${i}.mnc
    concat_list="$concat_list /tmp/tmp${i}.mnc "
done
mincconcat -clobber -2 -nocheck_dimensions $concat_list output/srv/srv_space-rec_nl.mnc

concat_list=""
for i in `seq 1 6`; do
    mincresample -clobber -coronal output/srv/srv_space-rec-${i}_lin.mnc /tmp/tmp${i}.mnc
    minc_modify_header -dinsert yspace:step=-1 /tmp/tmp${i}.mnc
    concat_list="$concat_list /tmp/tmp${i}.mnc "
done
mincconcat -clobber -2 -nocheck_dimensions $concat_list output/srv/srv_space-rec_lin.mnc

concat_list=""
for i in `seq 1 6`; do
    concat_list="$concat_list output/srv/vol_cls_${i}_morph_blur_bin_blur_rsl_space-slab-${i}.mnc "
done
minccalc -clobber -expr "sum(A)" $concat_list output/srv/vol_cls_nl.mnc

#concat_list=""
#for i in `seq 1 6`; do
#    concat_list="$concat_list output/srv/vol_${i}_in_blur_rsl_space-slab-${i}.mnc"
#done
#minccalc -clobber -expr "sum(A)" $concat_list output/srv/vol_nl.mnc

exit 0
mkdir -p output/coronal output/coronal/qc
rm /tmp/log.txt
for i in `seq 1 6`; do
    srv=output/srv/srv_space-rec-${i}.mnc
    cls=../MR1/volume/vol_cls_${i}_morph_blur_bin.mnc
    yspace=`mincinfo -dimlength yspace $cls`
    
#    for y in `seq 39 $yspace`; do
    concat_list=""
    clobber=1
    for y in `seq 100 300`; do
        out="output/coronal/cls_${i}_${y}.mnc"
        qc="output/coronal/qc/qc_${i}_${y}.png"
        tfm1=output/coronal/tfm_rec-to-cls_1_${y}.xfm
        tfm2=output/coronal/tfm_rec-to-cls_2_${y}.xfm
        tfm3=output/coronal/tfm_rec-to-cls_3_${y}.xfm

        if [[ ! -f $out.gz || "$clobber" == 1 ]]; then
            y1=`echo "$y + 20" | bc` 
            mincreshape -clobber -dimrange yspace=${y},1 $srv /tmp/srv.mnc #&>> /tmp/log.txt
            mincreshape -clobber -dimrange yspace=${y},1 $cls /tmp/cls.mnc &>> /tmp/log.txt
            #mincreshape -clobber -dimrange yspace=${y1},1 $srv /tmp/cls.mnc #&>> /tmp/log.txt

            srv_max=`mincstats -quiet -max /tmp/srv.mnc` 
            cls_max=`mincstats -quiet -max /tmp/cls.mnc` 

            #register /tmp/cls.mnc /tmp/srv.mnc
            if [[ ! -f output/coronal/cls_${y}.mnc.gz || ! -f $qc || "$clobber" == "1" ]]; then
                if [[ $srv_max -gt 0 && $cls_max -gt 0 ]]; then
                    clobber=1
                    if [[ ! -f $tfm3 || "$clobber" == 1 ]]; then
                        base="-clobber -source_mask /tmp/cls.mnc -model_mask /tmp/srv.mnc"
                        step1="-step 1 0 1 "
                        step2="-step 2 2 2 "
                        step3="-step 2 2 2 "
                        s1="1 0 1"
                        s2="4 0 4"
                        s3="4 0 4"
                        mincblur -clobber  -3dfwhm $s1 -dimensions 2 /tmp/srv.mnc /tmp/srv
                        mincblur -clobber  -3dfwhm $s1 -dimensions 2 /tmp/cls.mnc /tmp/cls
                        minctracc  $base -est_translation $step1 -simplex 16 -lsq7  /tmp/cls_blur.mnc /tmp/srv_blur.mnc $tfm1
   
                        mincblur -clobber  -3dfwhm $s2 -dimensions 2 /tmp/srv.mnc /tmp/srv
                        mincblur -clobber  -3dfwhm $s2 -dimensions 2 /tmp/cls.mnc /tmp/cls
                        minctracc  $base -transformation $tfm1 $step2 -simplex 8 -lsq9  /tmp/cls_blur.mnc /tmp/srv_blur.mnc $tfm2
                        mincblur -clobber  -3dfwhm $s3 -dimensions 2 /tmp/srv.mnc /tmp/srv
                        mincblur -clobber  -3dfwhm $s3 -dimensions 2 /tmp/cls.mnc /tmp/cls
                        minctracc $base -transformation $tfm2  $step3 -simplex 4 -lsq12  /tmp/cls_blur.mnc /tmp/srv_blur.mnc $tfm3 
                        #register output/coronal/cls_${i}_${y}.mnc.gz /tmp/srv_blur.mnc
                        tfm=$tfm3
                    fi
                    clobber=1
                    mincresample -2 -nearest -clobber -transformation $tfm -like /tmp/srv.mnc /tmp/cls.mnc  /tmp/tmp.mnc &>> /tmp/log.txt

                    mincresample -coronal -2 -nearest /tmp/tmp.mnc $out &>> /tmp/log.txt
                    rm /tmp/tmp.mnc
                    gzip -f $out
                    
                    python3 srv_qc.py /tmp/cls.mnc /tmp/srv.mnc $out.gz $qc
                    
                    #register $out.gz /tmp/srv.mnc 
                else 
                    cp $cls $out
                    gzip  $out
                fi
            fi

        fi
        concat_list="$concat_list $out.gz"
    done
    mincconcat -clobber -2 -nocheck_dimensions $concat_list /tmp/tmp.mnc
    mincresample -clobber -2 -transverse /tmp/tmp.mnc output/coronal/slab_${i}.mnc
    rm /tmp/tmp.mnc

    gzip -f output/coronal/slab_${i}.mnc
    exit 0
done
