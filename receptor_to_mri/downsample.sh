downsample() {
        in=$1
        out=$2
        s=$3
        fwhm_1=$4
        clobber_1=$5
        if [[ ! -f $out || $clobber_1 == 1 ]]; then
            mincblur -quiet -clobber -fwhm $fwhm_1 $in /tmp/tmp
            z=`mincinfo -dimlength zspace $in`
            y=`mincinfo -dimlength yspace $in`
            x=`mincinfo -dimlength xspace $in`
            zstep=`mincinfo -attvalue zspace:step $in`
            ystep=`mincinfo -attvalue yspace:step $in`
            xstep=`mincinfo -attvalue xspace:step $in`
            zstart=`mincinfo -attvalue zspace:start $in`
            ystart=`mincinfo -attvalue yspace:start $in`
            xstart=`mincinfo -attvalue xspace:start $in`

            zw=`echo "(($z * $zstep)  ) " | bc`
            yw=`echo "(($y * $ystep)  ) " | bc`
            xw=`echo "(($x * $xstep)  ) " | bc`

            zmax=`echo "$zw / $s " | bc`
            ymax=`echo "$yw / $s " | bc`
            xmax=`echo "$xw / $s " | bc`
            step="$s $s $s"
            mincresample -quiet -clobber -step $step -nelements $xmax $ymax $zmax  /tmp/tmp_blur.mnc $out
        fi
        return 0
}


downsample $1 $2 $3 $4 $5 

