cls=`./check_file.sh $1`
srv=`./check_file.sh $2`
srv_rsl=$3
cls_rsl=$4
tfm_lin=$5
srv_lin=$6
clobber=$7

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
  

