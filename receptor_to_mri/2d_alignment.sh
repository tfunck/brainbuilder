weights="-w_translations 1 1 0 -w_rotations 0.0 0.0 0.0174533 -w_scales 0.02 0.02 0.0 -w_shear  0.20 0.0 0.0"

qdir="/home/t/neuro/projects/Juelich-Receptor-Atlas/receptor_to_mri/quarantines/Linux-x86_64/"
source ${qdir}init.sh
mkdir -p output/coronal output/coronal/qc
rm /tmp/log.txt
itr=${1:-"0"}
clobber=${2:-0}
ITR="_itr-${itr}"
for i in `seq 1 6`; do
    srv0="output/${ITR}/srv_itr-${itr}_space-rec-${i}_lin.mnc"
    cls0=../MR1/volume/vol_cls_${i}_morph_blur_bin.mnc
    yspace=`mincinfo -dimlength yspace $cls0`
    cls="../MR1/volume/vol_cls_${i}_morph_blur_bin_blur.mnc"
    srv="output/${ITR}/srv_itr-${itr}_space-rec-${i}_lin_blur.mnc"
    valid_slices=valid_cls_slices_${i}.txt 
    if [[ ! -f $cls || $clobber == 1 ]] ; then
        mincblur -quiet -clobber -3dfwhm 0 5 0 -dimensions 2 $cls0 ../MR1/volume/vol_cls_${i}_morph_blur_bin
    fi
    if [[ ! -f $srv || $clobber == 1 ]]; then
        mincblur -quiet -clobber -3dfwhm 0 5 0 -dimensions 2 $srv0 output/${ITR}/srv_itr-${itr}_space-rec-${i}_lin
    fi

    if [[ ! -f  $valid_slices || $clobber == 1 ]]; then
        for y in `seq 0 $yspace`; do
            ./extract.sh $y /tmp/cls_slice.mnc $cls0 0 0 $last_frame 
            cls_max=`mincstats -quiet -max /tmp/cls_slice.mnc` 
            if [[  $cls_max !=  "0" ]]; then
                echo $y >> $valid_slices
            fi
        done
    fi
    
    clobber=0 
    concat_list=""
    for y in `cat $valid_slices | xargs echo`; do
        out="output/coronal/cls_${i}_${y}.mnc"
        qc="output/coronal/qc/qc_${i}_${y}.png"
        tfm1=output/coronal/tfm_rec-to-cls_1_${y}.xfm
        tfm2=output/coronal/tfm_rec-to-cls_2_${y}.xfm
        tfm3=output/coronal/tfm_rec-to-cls_3_${y}.xfm
        concat_list="$concat_list $out.gz"
        if [[ ! -f $out.gz || "$clobber" == 1 ]]; then
            offset=0
            clobber=0
            if [[ ! -f $qc || "$clobber" == "1" ]]; then
                
                if [[ ! -f $tfm3 || "$clobber" == 1 ]]; then
                    ./extract.sh $y /tmp/cls_slice.mnc $cls0 0 0 $last_frame 
                    ./extract.sh $y /tmp/srv.mnc $srv 0 0 $last_frame
                    ./extract.sh $y /tmp/cls.mnc $cls 0 0 $last_frame
                    #register /tmp/cls.mnc /tmp/srv.mnc
                    base="-clobber $weights -source_mask /tmp/cls.mnc -model_mask /tmp/srv.mnc"
                    step1="-step 2 2 0 "
                    step2="-step 1 1 0 "
                    step3="-step 0.2 0.2 0 "
                    s1="2 2 0"
                    s2="1 1 0"
                    s3="0.5 0.5 0"

                    mincblur -quiet -clobber  -3dfwhm $s1 -dimensions 2 /tmp/srv.mnc /tmp/srv
                    mincblur -quiet -clobber  -3dfwhm $s1 -dimensions 2 /tmp/cls.mnc /tmp/cls
                    minctracc  $base -identity $step1 -simplex 2 -lsq6  /tmp/cls_blur.mnc /tmp/srv_blur.mnc $tfm1
                    #register /tmp/srv_blur.mnc /tmp/cls_blur.mnc
                    mincblur -quiet -clobber  -3dfwhm $s2 -dimensions 2 /tmp/srv.mnc /tmp/srv
                    mincblur -quiet -clobber  -3dfwhm $s2 -dimensions 2 /tmp/cls.mnc /tmp/cls
                    minctracc -quiet   $base -transformation $tfm1 $step2 -simplex 1 -lsq6  /tmp/cls_blur.mnc /tmp/srv_blur.mnc $tfm2
                    mincblur -quiet -clobber -3dfwhm $s3 -dimensions 2 /tmp/srv.mnc /tmp/srv
                    mincblur -quiet -clobber -3dfwhm $s3 -dimensions 2 /tmp/cls.mnc /tmp/cls
                    minctracc -quiet $base -transformation $tfm2  $step3 -simplex 0.5 -lsq6  /tmp/cls_blur.mnc /tmp/srv_blur.mnc $tfm3 
                    tfm=$tfm1
                fi
                mincresample -quiet -2 -nearest -clobber -transformation $tfm -like /tmp/srv.mnc /tmp/cls_slice.mnc  /tmp/tmp.mnc #&>> /tmp/log.txt
                #mincresample -clobber -coronal -2 -nearest /tmp/tmp.mnc $out #&>> /tmp/log.txt
                #rm /tmp/tmp.mnc
                cp /tmp/tmp.mnc $out
                #register $out /tmp/srv.mnc 
                gzip -f $out
                
                #register $out /tmp/srv_blur.mnc 
                #python3 srv_qc.py /tmp/cls.mnc /tmp/srv.mnc $out.gz $qc
            #else 
            #    echo hello
            #    exit 1
            #    cp $cls $out
            #    gzip  $out
            fi
        fi


        done
    echo mincconcat -clobber -2 -nocheck_dimensions $concat_list /tmp/tmp.mnc
    mincconcat -clobber -2 -nocheck_dimensions $concat_list /tmp/tmp.mnc
    echo mincresample -clobber -2 -transverse /tmp/tmp.mnc output/coronal/slab_${i}.mnc
    mincresample -clobber -2 -transverse /tmp/tmp.mnc output/coronal/slab_${i}.mnc
    rm /tmp/tmp.mnc

    gzip -f output/coronal/slab_${i}.mnc
    exit 0
done
