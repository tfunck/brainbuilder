downsample() {
        vol=$1
        s=$2
        fwhm_1=$3
        blur_rsl=`echo $vol | sed 's/.gz//'| sed 's/.mnc/_blur_rsl.mnc/'`
        mincblur -clobber -fwhm $fwhm_1 $vol /tmp/tmp
            z=`mincinfo -dimlength zspace $vol`
            y=`mincinfo -dimlength yspace $vol`
            x=`mincinfo -dimlength xspace $vol`
            zstep=`mincinfo -attvalue zspace:step $vol`
            ystep=`mincinfo -attvalue yspace:step $vol`
            xstep=`mincinfo -attvalue xspace:step $vol`
            zw=`echo "(($z * $zstep)  ) " | bc`
            yw=`echo "(($y * $ystep)  ) " | bc`
            xw=`echo "(($x * $xstep)  ) " | bc`
            zmax=`echo "$zw / $s " | bc`
            ymax=`echo "$yw / $s " | bc`
            xmax=`echo "$xw / $s " | bc`
            step="$s $s $s"
            mincresample -clobber -step $step -nelements $xmax $ymax $zmax /tmp/tmp_blur.mnc $4

        return 0
}

slab=3
concat_list=`cat slab_3_concat_list.txt | xargs echo`
concat_list_orig=`cat slab_3_concat_list.txt | sed 's/ /\n/g' | grep orig | xargs echo `
rec="../MR1/volume/vol_${slab}.mnc"
cls="../MR1/volume/vol_cls_${slab}.mnc"
srv="output/srv/srv_space-rec-${slab}_lin_20u.mnc"

if [[ ! -f fill/flum_3.mnc ]]; then
  python3 create_volume.py $concat_list $srv fill/flum_3.mnc
fi

if [[ ! -f fill/flum_orig_3.mnc ]]; then
  python3 create_volume.py $concat_list_orig $srv fill/flum_orig_3.mnc
fi
#register fill/flum_3.mnc fill/flum_orig_3.mnc

if [[ ! -f  fill/flum_3_1mm.mnc ]]; then
  downsample fill/flum_3.mnc 1 1   fill/flum_3_1mm.mnc
fi

python3 kmeans_vol.py  fill/flum_3_1mm.mnc 100  fill/flum_3_1mm_kmeans.mnc
