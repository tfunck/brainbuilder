weights="-w_translations 1 1 0 -w_rotations 0.0 0.0 0.0174533 -w_scales 0.02 0.02 0.0 -w_shear  0.20 0.0 0.0"
init="-est_translations"
qdir="/home/t/neuro/projects/Juelich-Receptor-Atlas/receptor_to_mri/quarantines/Linux-x86_64/"
#source ${qdir}init.sh
fn="../MR1/coregistration/receptor_slices.csv"
mkdir -p fill

extract(){
    #1==_y0
    #2==input filename
    #3==3D file
    echo $1 $2 $3
    mincreshape -clobber -dimrange yspace=${1},1 $3 $2 &> /dev/null
    minc_modify_header -dinsert yspace:start=0 $2 &> /dev/null
    ./flipyz.pl $2 &> /dev/null
    if [[ $4 == 1 ]]; then
        ${qdir}/bin/mincblur -clobber -no_apodize  -3dfwhm $S $S 0  -dimensions 2 $2 ${2%.*} &> /dev/null
    fi
}

align(){
    #Inputs
    #1=base slice number
    #2=s
    #3=target file
    #4=source file
    #5=rec
    #6=srv
    echo Align Inputs: $1 $2 $3 $4 $5 $6
    #Outputs
    xfm="fill/tfm_${1}_${2}.xfm"
    fill=fill/rec_slice-${1}_to_${2}.mnc
           
    extract ${1} /tmp/rec.mnc ${5}
    extract ${s} ${3} ${6}

    ${qdir}/bin/mincblur -clobber -3dfwhm $S $S 0  -dimensions 2 ${3} ${3%.*} #&> /dev/null
            
    minctracc -clobber $weights $init -xcorr -step 0.5 0.5 0 -simplex 1 -lsq12 -source_mask $4 -model_mask ${3} ${4%.*}_blur.mnc ${3%.*}"_blur.mnc" $xfm #&> /dev/null
    mincresample -clobber -transformation $xfm -like $3 /tmp/rec.mnc $fill #&> /dev/null
    
}

clobber=0
for i in `seq 3 3`; do
    rec="../MR1/volume/vol_${i}.mnc.gz"
    srv="output/srv/srv_space-rec-${i}_lin_20u.mnc"
    #srv=output/srv/mri1_gm_srv_space-slab-${i}_trunc_blur_rsl.mnc
    
    final_tfm=output/srv/srv_space-rec-${i}_lin+nl.tfm
    clobber=0 
    if [[ ! -f $srv || $clobber == 1 ]]; then
        #mincresample -nearest -clobber -transformation $final_tfm -like $rec output/srv/mri1_gm_srv_space-slab-${i}_trunc_blur_rsl.mnc $srv
        mincresample -nearest -clobber -transformation $final_tfm -like $rec  output/srv/mri1_gm_srv_space-slab-${i}_trunc.mnc $srv
    fi

    read -a y <<< `python3 get_ligand_slices.py $fn \"flum\" $i | grep Location | sed 's/Slice Location: //'`
    
    y1=`echo ${y[@]}`
    read -a y1 <<< `python3 -c "x=\"${y1}\".split(' '); print(' '.join(x[1:]))"`
    
    let "n = ${#y[@]} - 1"
    echo ${y1[@]}
    for i in `seq 0 $n`; do
        _y0=${y[$i]}   
        _y1=${y1[$i]}
        let "_start = 1 + $_y0"
        let "_end = $_y1 - 1"
        echo y0: $_y0 y1: $_y1

        S=1
       
        extract $_y0 /tmp/source.mnc $srv    
        
        seq_list=(`python -c "import numpy as np; print( \" \".join( np.flip(np.arange(${_start},${_y1},5),axis=0).astype(str) ))"`)
        inv_list=(`python -c "import numpy as np; print( \" \".join( np.arange(${_start},${_y1},5).astype(str)))"`)

        for s in ${seq_list[@]} ; do
            fill=fill/rec_slice-${_y0}_to_${s}.mnc
            if [[ ! -f $fill || $clobber == 1 ]]; then
                align $_y0 $s /tmp/target.mnc /tmp/source.mnc $rec $srv
            fi
        done
        
        extract $_y1 /tmp/source.mnc $srv 
        for s in ${inv_list[@]}; do
            fill=fill/rec_slice-${_y1}_to_${s}.mnc
            if [[ ! -f $fill || $clobber == 1 ]]; then
                align $_y1 $s /tmp/target.mnc /tmp/source.mnc $rec $srv 
            fi
        done
        let "n=${#seq_list[@]}-1"
        
    
        if [[ ! -f fill/rec_flum_${_y0}.mnc || $clobber == 1   ]] ; then
            extract $_y0 fill/rec_flum_${_y0}.mnc $rec 1    
        fi
        if [[ ! -f  fill/rec_flum_${_y1}.mnc || $clobber == 1 ]]; then
            extract $_y1 fill/rec_flum_${_y1}.mnc $rec 1 
        fi

        concat_list="fill/rec_flum_${_y0}.mnc "
        for i in `seq 0 $n`; do
            s0=${seq_list[$i]}
            #s1=${inv_list[$i]}
            img0=fill/rec_slice-${_y0}_to_${s0}.mnc
            img1=fill/rec_slice-${_y1}_to_${s0}.mnc
            x=`python -c "print( 1. * ($s0 - $_y0) / ($_y1 - $_y0 )  )"`
            if [[ ! -f fill/rec_flum_${s0}.mnc || $clobber == 1 ]]; then
                minccalc -clobber -expr "A[0]*$x + A[1]*(1-$x) " $img0 $img1 fill/rec_flum_${s0}.mnc
            fi
            concat_list="$concat_list fill/rec_flum_${s0}.mnc"
        done
        concat_list="$concat_list fill/rec_flum_${_y1}.mnc"
        echo $concat_list
        python3 create_volume.py $concat_list $rec fill/flum_${i}.mnc

        exit 1
    done 
done




