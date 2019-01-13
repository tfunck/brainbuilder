weights="-w_translations 1 1 0 -w_rotations 0.0 0.0 0.0174533 -w_scales 0.02 0.02 0.0 -w_shear  0.20 0.0 0.0"
init="-est_translations"
S=3
qdir="/home/t/neuro/projects/Juelich-Receptor-Atlas/receptor_to_mri/quarantines/Linux-x86_64/"
source ${qdir}init.sh
fn="../MR1/coregistration/receptor_slices.csv"
mkdir -p fill

align(){
    #Inputs
    #1=base slice number
    #2=s
    #3=target file
    #4=source file
    #5=rec slice
    #6=srv
    _fill=$7
    #Outputs
    xfm="fill/tfm_${1}_${2}.xfm"

    #echo extract.sh ${s} ${3} ${6} 1
    ./extract.sh ${s} ${3} ${6} 1
    #echo

    #echo ${qdir}/bin/mincblur -quiet -clobber -3dfwhm $S $S 0  -dimensions 2 ${3} ${3%.*}
    ${qdir}/bin/mincblur -quiet -clobber -3dfwhm $S $S 0  -dimensions 2 ${3} ${3%.*} > /dev/null
    #echo
    #echo minctracc -quiet -clobber $weights $init -xcorr -step 0.5 0.5 0 -simplex 1 -lsq12 -source_mask $4 -model_mask ${3} ${4%.*}"_blur.mnc" ${3%.*}"_blur.mnc" $xfm
    minctracc -quiet -clobber $weights $init -xcorr -step 0.5 0.5 0 -simplex 1 -lsq12 -source_mask $4 -model_mask ${3} ${4%.*}"_blur.mnc" ${3%.*}"_blur.mnc" $xfm > /dev/null


    #echo mincresample -quiet -clobber -transformation $xfm -like $3 $5 $_fill
    mincresample -quiet -clobber -transformation $xfm -tfm_input_sampling $5 $_fill > /dev/null
    minc_modify_header -dinsert yspace:start=`mincinfo -attvalue yspace:start /tmp/target.mnc` $_fill
    minc_modify_header -dinsert zspace:start=`mincinfo -attvalue zspace:start /tmp/target.mnc` $_fill
    minc_modify_header -dinsert xspace:start=`mincinfo -attvalue xspace:start /tmp/target.mnc` $_fill
    minc_modify_header -dinsert yspace:step=`mincinfo -attvalue yspace:step /tmp/target.mnc` $_fill
    minc_modify_header -dinsert zspace:step=`mincinfo -attvalue zspace:step /tmp/target.mnc` $_fill
    minc_modify_header -dinsert xspace:step=`mincinfo -attvalue xspace:step /tmp/target.mnc` $_fill
    #echo
}

#############################
# Improve CLS SRV alignment #
#############################
improve_cls_srv_alignment(){
_y0=$1
_slab=$2
_cls_slice=$3
_srv_slice=$4
_cls=$5
_srv=$6
_clobber=$7

    out_tfm="fill/tfm_${slab}_${_y0}.xfm"
    out_tfm_inv="fill/tfm_${slab}_${_y0}_inv.xfm"
    fill=fill/srv_rsl_${_slab}_${_y0}.mnc
    if [[ ! -f $_cls_slice || $_clobber == 1 ]]; then
        echo "Extract Classified Slice"
        ./extract.sh $_y0 $_cls_slice $_cls 1
    fi

    if [[ ! -f $_srv_slice || $_clobber == 1  ]]; then
        echo "Extract SRV Slice"
        ./extract.sh $_y0 $_srv_slice $_srv 1
    fi

    if [[ "`mincstats -quiet -sum $_srv_slice`" == "0" || "`mincstats -quiet -sum $_srv_slice`" == "0" ]]; then
      echo Returning `mincstats -quiet -sum $_srv_slice` `mincstats -quiet -sum $_srv_slice`
      return 1
    fi

    if [[ ! -f $out_tfm || ! -f $fill || $_clobber == 1 ]]; then
      minctracc -clob $init $weights -step 0.5 0.5 0 -lsq9 ${_srv_slice%.*}_blur.mnc ${_cls_slice%.*}_blur.mnc  $out_tfm
      mincresample -clob -transformation  $out_tfm -tfm_input_sampling $_srv_slice $fill
    fi

    if [[ ! -f $out_tfm_inv || $_clobber == 1 ]]; then
      echo xfminvert -clobber $out_tfm $out_tfm_inv
      xfminvert -clobber $out_tfm $out_tfm_inv
    fi

  #register $_cls_slice  $fill
}

########
# MAIN #
########
clobber=0
for slab in `seq 3 3`; do
    rec="../MR1/volume/vol_${slab}.mnc"
    srv="output/srv/srv_space-rec-${slab}_lin_20u.mnc"
    #srv="output/srv/mri1_gm_srv_space-slab-${slab}_trunc.mnc"
    cls="../MR1/volume/vol_cls_3_morph_blur_bin.mnc"
    #srv_nl="output/srv/srv_space-rec-${slab}_nl.mnc"
    #srv=output/srv/mri1_gm_srv_space-slab-${i}_trunc_blur_rsl.mnc
    srv_cls_tfm="output/srv/srv_space-rec-${slab}_lin+nl.tfm"

    if [[ ! -f $srv || $clobber == 1 ]]; then
        mincresample -nearest -clobber -transformation $srv_cls_tfm -like $cls  output/srv/mri1_gm_srv_space-slab-${slab}_trunc.mnc $srv
    fi

    python3 get_ligand_slices.py $fn \"flum\" $slab
    read -a y <<< `python3 get_ligand_slices.py $fn \"flum\" $slab | grep Location | sed 's/Slice Location: //'`
    y1=`echo ${y[@]}`
    read -a y1 <<< `python3 -c "x=\"${y1}\".split(' '); print(' '.join(x[1:]))"`

    echo ${y[@]}

    for _y0 in ${y[@]} ; do
      cls_slice="fill/cls_orig_${slab}_${_y0}.mnc"
      srv_slice="fill/slice_${slab}_${_y0}.mnc"
      echo $_y0 $slab $cls_slice $srv_slice $cls $srv $clobber
      improve_cls_srv_alignment $_y0 $slab $cls_slice $srv_slice $cls $srv $clobber
    done

    concat_list=""
    #y=("552" "622")
    #y1=("622")
    let "n = ${#y[@]} - 2"
    echo ${y[@]}
    echo ${y1[@]}

    for i in `seq 0 $n`; do
        _y0=${y[$i]}
        _y1=${y1[$i]}
        let "_start = 1 + $_y0"
        let "_end = $_y1 - 1"
        srv_y0="fill/slice_${slab}_${_y0}.mnc"
        srv_y1="fill/slice_${slab}_${_y1}.mnc"
        rec_y0="fill/rec_orig_${slab}_${_y0}.mnc"
        rec_y1="fill/rec_orig_${slab}_${_y1}.mnc"
        echo y0: $_y0 y1: $_y1

        tfm_y0="fill/tfm_${slab}_${_y0}_inv.xfm"
        tfm_y1="fill/tfm_${slab}_${_y1}_inv.xfm"

        #######################
        # Extract base slices #
        #######################
        if [[ ! -f  $srv_y0 || $clobber == 1 ]]; then
            ./extract.sh $_y0 $srv_y0 $srv 1
        fi
        if [[ ! -f  $srv_y1 || $clobber == 1 ]]; then
            ./extract.sh $_y1 $srv_y1 $srv 1
        fi
        if [[ ! -f  $rec_y0 || $clobber == 1 ]]; then
            ./extract.sh $_y0 $rec_y0 $rec 0 $tfm_y0
        fi
        if [[ ! -f  $srv_y1 || $clobber == 1 ]]; then
            ./extract.sh $_y1 $rec_y1 $rec 0  $tfm_y1
        fi

        step=1
        seq_list=(`python -c "import numpy as np; print( \" \".join( np.flip(np.arange(${_start},${_y1},${step}),axis=0).astype(str) ))"`)
        inv_list=(`python -c "import numpy as np; print( \" \".join( np.arange(${_start},${_y1},${step}).astype(str)))"`)

        echo Aligning from $_y0 $_y1
        for s in ${seq_list[@]} ; do
            fill="fill/rec_slice-${_y0}_to_${s}.mnc"

            if [[ ! -f $fill || $clobber == 1 ]]; then
                echo $_y0 --> $s
                align $_y0 $s /tmp/target.mnc $srv_y0 $rec_y0 $srv $fill
            fi
        done

        echo Aligning from $_y1 $_y0
        for s in ${inv_list[@]}; do
            fill=fill/rec_slice-${_y1}_to_${s}.mnc
            #if [[ ! -f $fill || $clobber == 1 ]]; then
            #    ./extract.sh $s $fill $srv
            #fi
            if [[ ! -f $fill || $clobber == 1 ]]; then
                echo $_y1 --> $s
                align $_y1 $s /tmp/target.mnc $srv_y1 $rec_y1 $srv $fill
            fi
        done

        let "n0=${#seq_list[@]}-1"

        concat_list="$concat_list $rec_y0 "
        for ii in `seq 0 $n0`; do
            s0=${seq_list[$ii]}
            img0="fill/rec_slice-${_y0}_to_${s0}.mnc"
            img1="fill/rec_slice-${_y1}_to_${s0}.mnc"
            x=`python -c "print( 1. * ($s0 - $_y0) / ($_y1 - $_y0 )  )"`
            #if [[ "$x" > "0.5" ]] ; then
            #    cp $img1 fill/rec_flum_${s0}.mnc
            #else
            #    cp $img0 fill/rec_flum_${s0}.mnc
            #fi
            minc_modify_header -dinsert xspace:direction_cosines=1,0,0 -dinsert yspace:direction_cosines=0,1,0  -dinsert zspace:direction_cosines=0,0,1 $img0
            minc_modify_header -dinsert xspace:direction_cosines=1,0,0 -dinsert yspace:direction_cosines=0,1,0  -dinsert zspace:direction_cosines=0,0,1 $img1
            clobber=1
            if [[ ! -f fill/rec_flum_${s0}.mnc || $clobber == 1 ]]; then
                echo fill/rec_flum_${s0}.mnc
                minccalc -quiet -clobber -expr "A[0]*(1-$x) + A[1]*$x " $img0 $img1 fill/rec_flum_${s0}.mnc
            fi
            clobber=0
            concat_list="$concat_list fill/rec_flum_${s0}.mnc"

        done
        concat_list="$concat_list $rec_y1"
        #if [[ $_y1 == 622 ]]; then
        #    break
        #fi

    done
    echo $concat_list > slab_${slab}_concat_list.txt

done
