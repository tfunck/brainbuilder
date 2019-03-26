
extract(){
    #1==_y0
    #2==input filename
    #3==3D file
    vol=$3
    _end=${6:-1}
    gz=0

    mincreshape -quiet -clobber -dimrange yspace=${1},$_end $vol $2 #> /dev/null
    if [[ -f $5 ]]; then
        mincresample -quiet -clob -transformation $5 -tfm_input_sampling $2 /tmp/tmp.mnc
    fi
    #minc_modify_header -dinsert yspace:start=0 $2 > /dev/null
    start=`mincinfo -attvalue yspace:start $vol`
    step=`mincinfo -attvalue yspace:step $vol`
    new_start=`python -c "print($start + $step * $1)"`
    
    minc_modify_header -dinsert yspace:start=${new_start}  $2
    #./flipyz.pl $2 > /dev/null
    mincreshape -clobber -dimrange yspace=0,0 $2 /tmp/tmp.mnc
    mv /tmp/tmp.mnc $2
    if [[ $4 == 1 ]]; then
        ${qdir}/bin/mincblur -quiet -clobber -no_apodize  -3dfwhm $S $S 0  -dimensions 2 $2 ${2%.*} > /dev/null
    fi

}

extract $1 $2 $3 $4 $5 $6
