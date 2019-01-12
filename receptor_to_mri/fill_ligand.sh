weights="-w_translations 1 1 0 -w_rotations 0.0 0.0 0.0174533 -w_scales 0.02 0.02 0.0 -w_shear  0.20 0.0 0.0"
init="-est_translations"
S=1
qdir="/home/t/neuro/projects/Juelich-Receptor-Atlas/receptor_to_mri/quarantines/Linux-x86_64/"
source ${qdir}init.sh
fn="../MR1/coregistration/receptor_slices.csv"
mkdir -p fill

extract(){
    #1==_y0
    #2==input filename
    #3==3D file
    mincreshape -quiet -clobber -dimrange yspace=${1},1 $3 $2 > /dev/null
    minc_modify_header -dinsert yspace:start=0 $2 > /dev/null
    ./flipyz.pl $2 > /dev/null
    if [[ $4 == 1 ]]; then
        ${qdir}/bin/mincblur -quiet -clobber -no_apodize  -3dfwhm $S $S 0  -dimensions 2 $2 ${2%.*} > /dev/null
    fi
}

align(){
    #Inputs
    #1=base slice number
    #2=s
    #3=target file
    #4=source file
    #5=rec slice
    #6=srv
    #Outputs
    xfm="fill/tfm_${1}_${2}.xfm"
    fill=fill/rec_slice-${1}_to_${2}.mnc
           
    extract ${s} ${3} ${6} 1

    ${qdir}/bin/mincblur -quiet -clobber -3dfwhm $S $S 0  -dimensions 2 ${3} ${3%.*} > /dev/null
    minctracc -quiet -clobber $weights $init -xcorr -step 0.5 0.5 0 -simplex 1 -lsq12 -source_mask $4 -model_mask ${3} ${4%.*}_blur.mnc ${3%.*}"_blur.mnc" $xfm > /dev/null
    mincresample -quiet -clobber -transformation $xfm -like $3 $5 $fill > /dev/null
}

clobber=0
for slab in `seq 3 3`; do
    rec="../MR1/volume/vol_${slab}.mnc"
    #srv="output/srv/srv_space-rec-${slab}_lin_20u.mnc"
    srv="output/srv/mri1_gm_srv_space-slab-${slab}_trunc.mnc"

    cls="../MR1/volume/vol_cls_${slab}.mnc"
    srv_nl="output/srv/srv_space-rec-${slab}_nl.mnc"
    
    #srv=output/srv/mri1_gm_srv_space-slab-${i}_trunc_blur_rsl.mnc
    
    final_tfm=output/srv/srv_space-rec-${slab}.tfm
    clobber=0 
    if [[ ! -f $srv || $clobber == 1 ]]; then
        mincresample -quiet -nearest -clobber -transformation $final_tfm -like $rec  output/srv/mri1_gm_srv_space-slab-${slab}_trunc.mnc $srv
    fi

    read -a y <<< `python3 get_ligand_slices.py $fn \"flum\" $slab | grep Location | sed 's/Slice Location: //'`
    
    y1=`echo ${y[@]}`
    read -a y1 <<< `python3 -c "x=\"${y1}\".split(' '); print(' '.join(x[1:]))"`
    
    echo ${y[@]}

    for _y0 in ${y[@]} ; do
        clobber=1
        if [[ ! -f fill/cls_orig_${_y0}.mnc || $clobber == 1 ]];then
            echo "Extract Classified Slice"
            #extract $_y0 fill/cls_orig_${slab}_${_y0}.mnc $cls 1
            mincreshape -quiet -clobber -dimrange yspace=${_y0},1 $cls fill/cls_orig_${slab}_${_y0}.mnc
            start=`mincinfo -attvalue yspace:start ../MR1/volume/vol_cls_${slab}.mnc`
            step=`mincinfo -attvalue yspace:step ../MR1/volume/vol_cls_${slab}.mnc`
            new_start=`python -c "print($start + $step * $_y0)"`
            echo $new_start = $start $step $_y0
            minc_modify_header -dinsert yspace:start=${new_start}  fill/cls_orig_${slab}_${_y0}.mnc
            #register output/srv/srv_space-rec-${slab}_lin.mnc  fill/cls_orig_${slab}_${_y0}.mnc
        fi

        if [[ ! -f  fill/slice_${slab}_${_y0}.mnc || $clobber == 1  ]]; then
            echo "Extract SRV Slice"
            #extract $_y0 fill/slice_${slab}_${_y0}.mnc $srv 1
            mincreshape -quiet -clobber -dimrange yspace=${_y0},1 $srv  fill/slice__${slab}_${_y0}.mnc 
        fi       

        if [[ ! -f  fill/rec_orig_${slab}_${_y0}.mnc || $clobber == 1  ]]; then
            echo "Extract receptor Slice"
            extract $_y0 fill/rec_orig_${slab}_${_y0}.mnc $rec 
        fi

        if [[ ! -f fill/cls_orig_${slab}_${_y0}.xfm || ! -f fill/slice_${slab}_${_y0}.mnc || $clobber == 1 ]]; then
            #mincresample -clob -invert_transformation -transformation output/srv/srv_space-rec-${slab}.tfm -tfm_input_sampling fill/cls_orig_${slab}_${_y0}.mnc /tmp/temp.mnc
            
            register fill/cls_orig_${slab}_${_y0}.mnc fill/slice_${slab}_${_y0}.mnc 
            exit 1
            minctracc -clob $weights -transformation output/srv/srv_space-rec-${slab}.tfm -lsq12 -model_mask fill/slice_${slab}_${_y0}.mnc fill/cls_orig_${slab}_${_y0}.mnc fill/slice_${slab}_${_y0}.mnc fill/cls_orig_${slab}_${_y0}.xfm
            mincresample -clob -transformation fill/cls_orig_${slab}_${_y0}.xfm -like fill/slice_${slab}_${_y0}.mnc fill/cls_orig_${slab}_${_y0}.mnc fill/cls_rsl_${slab}_${_y0}.mnc
        fi
        clobber=0
        register fill/slice_${slab}_${_y0}.mnc  fill/cls_rsl_${slab}_${_y0}.mnc
        exit 1
    done
    exit 0 


    concat_list=""
    let "n = ${#y[@]} - 2"
    for i in `seq 0 $n`; do
        _y0=${y[$i]}   
        _y1=${y1[$i]}
        let "_start = 1 + $_y0"
        let "_end = $_y1 - 1"
        echo y0: $_y0 y1: $_y1
      
        if [[ ! -f  fill/slice_${slab}_${_y0}.mnc ]]; then
            extract $_y0 fill/slice_${slab}_${_y0}.mnc $srv   
        fi
        if [[ ! -f  fill/slice_${slab}_${_y1}.mnc ]]; then
            extract $_y1 fill/slice_${slab}_${_y1}.mnc $srv    
        fi
        step=1
        seq_list=(`python -c "import numpy as np; print( \" \".join( np.flip(np.arange(${_start},${_y1},${step}),axis=0).astype(str) ))"`)
        inv_list=(`python -c "import numpy as np; print( \" \".join( np.arange(${_start},${_y1},${step}).astype(str)))"`)

        for s in ${seq_list[@]} ; do
            fill=fill/rec_slice-${_y0}_to_${s}.mnc
            if [[ ! -f $fill || $clobber == 1 ]]; then
                echo $_y0 --> $s 
                align $_y0 $s /tmp/target.mnc /tmp/source.mnc fill/rec_orig_${slab}_${_y0}.mnc $srv
            fi
        done
        
        for s in ${inv_list[@]}; do
            fill=fill/rec_slice-${_y1}_to_${s}.mnc
            if [[ ! -f $fill || $clobber == 1 ]]; then
                echo $_y1 --> $s
                align $_y1 $s /tmp/target.mnc /tmp/source.mnc fill/rec_orig_${slab}_${_y0}.mnc $srv 
            fi

        done
        
        let "n0=${#seq_list[@]}-1"
    
        if [[ ! -f fill/rec_flum_${_y0}.mnc || $clobber == 1   ]] ; then
            extract $_y0 fill/rec_flum_${_y0}.mnc $rec 1    
        fi
        if [[ ! -f  fill/rec_flum_${_y1}.mnc || $clobber == 1 ]]; then
            extract $_y1 fill/rec_flum_${_y1}.mnc $rec 1 
        fi

        concat_list="$concat_list fill/rec_flum_${_y0}.mnc "
        for ii in `seq 0 $n0`; do
            s0=${seq_list[$ii]}
            img0=fill/rec_slice-${_y0}_to_${s0}.mnc
            img1=fill/rec_slice-${_y1}_to_${s0}.mnc
            x=`python -c "print( 1. * ($s0 - $_y0) / ($_y1 - $_y0 )  )"`
            if [[ ! -f fill/rec_flum_${s0}.mnc || $clobber == 1 ]]; then
                minccalc -quiet -clobber -expr "A[0]*$x + A[1]*(1-$x) " $img0 $img1 fill/rec_flum_${s0}.mnc
            fi
            concat_list="$concat_list fill/rec_flum_${s0}.mnc"
        done
        concat_list="$concat_list fill/rec_flum_${_y1}.mnc"
        #if [[ $_y1 == 622 ]]; then
        #    break
        #fi
    done 
    echo $concat_list 
    python3 create_volume.py $concat_list $rec fill/flum_${slab}.mnc
done




