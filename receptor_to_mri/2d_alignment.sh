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
