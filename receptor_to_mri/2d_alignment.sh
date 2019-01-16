set_valid_slices(){
    _valid_slices=$1
    _yspace=$2
    _cls0=$3
    _clobber=$4
    if [[ ! -f  $_valid_slices || $_clobber == 1 ]]; then
        for _y in `seq 0 $_yspace`; do
            ./extract.sh $_y /tmp/cls_slice.mnc $_cls0
            cls_max=`mincstats -quiet -max /tmp/cls_slice.mnc` 
            if [[  $cls_max !=  "0" ]]; then
                echo $_y >> $valid_slices
                cp /tmp/cls_slice.mnc "output/${ITR}/${_y}_"`basename $_cls0`
                #Display  "output/${ITR}/${_y}_"`basename $_cls0`
                echo  "${out_dir}/${_y}_"`basename $_cls0`

            fi
        done
    fi
}

cls0=`./check_file.sh ${1}`
cls=`./check_file.sh ${2}`
srv=`./check_file.sh ${3}`
valid_slices=${4}
out_dir=$5
cls_rsl=$6
clobber=$7


if [[ ! -f $cls_rsl || $clobber == 1 ]]; then
    if [[ "${cls0##*.}" == "gz" ]]; then
        gunzip -kf $cls0
        cls0=${cls0%.*}
    fi

    if [[ "${cls##*.}" == "gz" ]]; then
        gunzip -kf $cls
        cls=${cls%.*}
    fi

    if [[ "${srv##*.}" == "gz" ]]; then
        gunzip -kf $srv
        srv=${srv%.*}
    fi
    
    printf "\t2D Alignment\n"
    let "yspace = `mincinfo -dimlength yspace $cls0` - 1"
    #yspace=`mincheader $cls0 | grep yspace:length | awk '{split($0,ar," "); print ar[3]}'`
    out_dir=${out_dir}"/2D"
    mkdir -p ${out_dir}
    #set_valid_slices $valid_slices $yspace $cls0  $clobber
    concat_list=""
    clobber=0
    for y in `seq 0 $yspace`; do
        out="${out_dir}/cls_${y}.mnc"
        tfm=${out_dir}/tfm_rec-to-cls_${y}.xfm
        cls_slice=${out_dir}/${y}_`basename $cls0`
        cls_blur_slice=${out_dir}/${y}_`basename $cls`
        srv_blur_slice=${out_dir}/${y}_`basename $srv`
        
        if [[ ! -f $cls_slice  || $clobber == 1 ]]; then
            echo  ./extract.sh $y $cls_slice $cls0
            ./extract.sh $y $cls_slice $cls0
        fi
         
        if [[ `grep -w $y $valid_slices` ]] ; then
            if [[ ! -f $out.gz ||  ! -f $tfm || "$clobber" == 1 ]]; then
                #if [[ ! -f $cls_slice  || $clobber == 1 ]]; then
                #    ./extract.sh $y $cls_slice $cls0
                #fi
                if [[ ! -f $srv_blur_slice || $clobber == 1 ]]; then
                    ./extract.sh $y $srv_blur_slice $srv
                fi
                ./bestlinreg_2d.sh $cls_slice $srv_blur_slice $tfm
            fi
            mincresample -quiet -2 -nearest -clobber -transformation $tfm -use_input_sampling $cls_slice $out
        else 
            #if [[ ! -f $srv_blur_slice || $clobber == 1 ]]; then
            #    ./extract.sh $y $srv_blur_slice $srv
            #fi
            cp $cls_slice $out
        fi

        gzip -f $out
        if [[ -f $out.gz ]]; then
            concat_list="$concat_list $out.gz"
        fi
        #if [[ $y == 234 ]] ; then
        #    exit 1
        #    break
        #fi
    done

    python2 concat.py $cls0 $concat_list $cls_rsl #> ./tmp.txt
fi

if [[ "${1##*.}" == "gz" && "${cls0##*.}" != "gz" ]]; then
    rm $cls0
fi

if [[ "${2##*.}" == "gz"  && "${cls##*.}" != "gz" ]]; then
    rm $cls
fi

if [[ "${3##*.}" == "gz"  && "${srv##*.}" != "gz" ]]; then
    rm $srv
fi


