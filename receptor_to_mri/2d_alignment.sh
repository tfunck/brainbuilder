set_valid_slices(){
    _valid_slices=$1
    _yspace=$2
    _cls0=$3
    _srv=$4
    _clobber=$5
    if [[ ! -f  $_valid_slices || $_clobber == 1 ]]; then
        for _y in `seq 0 $_yspace`; do
            cls_slice=${out_dir}/${y}_`basename $_cls0`
            srv_slice=${out_dir}/${y}_`basename $_srv`
            echo "2D $_y"
            ./extract.sh $_y $cls_slice $_cls0
            ./extract.sh $_y $srv_slice $_srv
            cls_max=`mincstats -quiet -max $cls_slice` 
            srv_max=`mincstats -quiet -max $srv_slice` 
            if [[  $cls_max !=  "0" && $srv_max != "0" ]]; then
                echo $_y >> $valid_slices
                cp /tmp/cls_slice.mnc "${out_dir}/${_y}_"`basename $_cls0`
                echo "${out_dir}/${_y}_"`basename $_cls0`

            fi
        done
    fi
}

cls0=${1}
cls=${2}
srv=${3}
valid_slices=${4}
out_dir=$5
cls_rsl=$6
clobber=$7
if [[ ! -f  ${cls_rsl%.*} || $clobber == 1 ]]; then
    if [[ "${cls0##*.}" == "gz" ]]; then
        gunzip -kf $cls0
        cls0=${cls0%.*}
    fi

    if [[ "${cls##*.}" == "gz" ]]; then
        gunzip -kf $cls
        cls=${cls%.*}
    fi

    #if [[ "${srv##*.}" == "gz" ]]; then
    #    gunzip -kf $srv
    #    srv=${srv%.*}
    #fi

    srv_rsl=`echo $srv | sed 's/.mnc/_rsl.mnc/g'`
    if [[ ! -f $srv_rsl || $clobber == 1 ]]; then
        echo mincresample -clobber $srv -like $cls $srv_rsl 
        mincresample -quiet -clobber $srv -like $cls $srv_rsl 
    fi
    srv=$srv_rsl
    printf "\t2D Alignment\n"
    let "yspace = `mincinfo -dimlength yspace $cls0` - 1"
    #yspace=`mincheader $cls0 | grep yspace:length | awk '{split($0,ar," "); print ar[3]}'`
    out_dir=${out_dir}"/2D"
    mkdir -p ${out_dir}
    
    set_valid_slices $valid_slices $yspace $cls0 $srv $clobber
    concat_list=""
    for y in `seq 0 $yspace`; do
        echo $y
        out="${out_dir}/cls_${y}.mnc"
        out_nii="${out_dir}/cls_${y}.nii"
        tfm=${out_dir}/tfm_rec-to-cls_${y}.xfm
        cls_slice=${out_dir}/${y}_`basename $cls0`
        srv_blur_slice=${out_dir}/${y}_`basename $srv`
        
        cls_slc_nii=`echo ${cls_slice%.*} | sed 's/.gz//'`.nii
        srv_slc_nii=`echo ${srv_blur_slice%.*} | sed 's/.gz//'`.nii

        if [[ ! -f $cls_slice  || $clobber == 1  ]]; then
            echo  ./extract.sh $y $cls_slice $cls0
            ./extract.sh $y $cls_slice $cls0
        fi
         
        if [[ `grep -w $y $valid_slices` ]] ; then
            if [[ ! -f $out || "$clobber" == 1 ]]; then
                if [[ ! -f $srv_blur_slice || $clobber == 1 ]]; then
                    ./extract.sh $y $srv_blur_slice $srv
                    echo ./extract.sh $y $srv_blur_slice $srv
                fi
                #./bestlinreg_2d.sh $cls_slice $srv_blur_slice $tfm
                mnc2nii $cls_slice $cls_slc_nii
                mnc2nii $srv_blur_slice $srv_slc_nii 
                mkdir -p $out_dir/${y} 
                 
                echo elastix -f $srv_slc_nii -m $cls_slc_nii -p Parameters_Rigid_2D.txt -out $out_dir/${y}/  > /dev/null
                elastix -f $srv_slc_nii -m $cls_slc_nii -p Parameters_Rigid_2D.txt -out $out_dir/${y}/  > /dev/null
                errorcode=$?
                 
                if [[ $errorcode != "0" ]]; then
                    nii2mnc -clobber $out_dir/${y}/result.0.nii $out
                    gzip -f $out
                fi
            fi
            #if [[ ! `grep -w $y $valid_slices` || ! -f $out ]] ; then
            #    cp $cls_slice $out
            #fi

        fi
        if [[ -f ${out_nii} ]]; then
            concat_list="$concat_list ${out_nii}"
        fi
        #if [[ $y == 234 ]] ; then
        #    exit 1
        #    break
        #fi
    done
	echo Concatenating
    python2 concat.py $cls0 $concat_list ${cls_rsl%.*} #> ./tmp.txt
    echo gzip -f ${cls_rsl%.*}
    gzip -f ${cls_rsl%.*}
fi

if [[ "${1##*.}" == "gz" && "${cls0##*.}" != "gz" ]]; then
    rm $cls0
fi

if [[ "${2##*.}" == "gz"  && "${cls##*.}" != "gz" ]]; then
    rm $cls
fi

#if [[ "${3##*.}" == "gz"  && "${srv##*.}" != "gz" ]]; then
#    rm $srv
#fi


