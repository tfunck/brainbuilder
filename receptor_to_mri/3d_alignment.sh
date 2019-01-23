itr=$1
cls=`./check_file.sh $2`
srv=`./check_file.sh $3`
srv_rsl=$4
cls_rsl=$5
srv_lin_rsl=$6
tfm_lin=$7
tfm_nl=$8
srv_lin=$9
srv_nl=${10}
clobber=${11}

########################
# DOWNSAMPLE CLS & SRV #
########################
./downsample.sh $srv $srv_rsl 0.5 0.5 $clobber
./downsample.sh $cls $cls_rsl 0.5 0.5 $clobber

####################
# LINEAR ALIGNMENT #
####################
if [[ ! -f $tfm_lin || "$clobber" -eq "1" ]] ; then
    printf "\t\tPerforming linear alignment of SRV to CLS..."
    ./bestlinreg_hires.sh $srv $cls $srv_rsl $cls_rsl $tfm_lin
    printf "\tDone.\n"
fi

if [[ ! -f $srv_lin || "$clobber" == 1 ]]; then
    mincresample -quiet -nearest -clobber -transformation $tfm_lin -like $cls $srv $srv_lin
fi
  
###########################
# 3D Non-linear Alignment #
###########################
if [[ "$itr" -gt 0 ]] ; then
    ./downsample.sh $srv_lin $srv_lin_rsl 0.5 0.5 $clobber
    if [[ ! -f $tfm_nl || "$clobber" -eq "1" ]] ; then
        printf "\t\tNon-linear alignment\n"
        ./bestnlreg_hires.sh $srv_lin $cls $srv_lin_rsl $cls_rsl /tmp/tmp_nl.xfm 
        xfmconcat -clob $tfm_lin /tmp/tmp_nl.xfm $tfm_nl
    fi

    if [[ ! -f $srv_nl || "$clobber" == 1 ]]; then
        mincresample -quiet -nearest -clobber -transformation $tfm_nl -like $cls $srv $srv_nl
    fi
else 
 cp $srv_lin $srv_nl
fi
