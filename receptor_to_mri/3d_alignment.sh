itr=$1
cls=$2
srv=$3
srv_rsl=$4
cls_rsl=$5
srv_lin_rsl=$6
tfm_lin=$7
tfm_nl=$8
srv_lin=$9
srv_nl=${10}
clobber=${11}
config=${12}

########################
# DOWNSAMPLE CLS & SRV #
########################
./downsample.sh $srv $srv_rsl 0.5 0.5 $clobber
./downsample.sh $cls $cls_rsl 0.5 0.5 $clobber

mnc2nii $srv ${srv%.*}.nii
mnc2nii $cls ${cls%.*}.nii
echo elastix -f ${cls%.*}.nii -m ${srv%.*}.nii -p Parameters_Affine.txt -p Parameters_BSpline.txt -out temp/
elastix -f ${cls%.*}.nii -m ${srv%.*}.nii -p Parameters_Affine.txt -p Parameters_BSpline.txt -out temp/

exit 0
####################
# LINEAR ALIGNMENT #
####################
if [[ ! -f $tfm_lin || "$clobber" -eq "1" ]] ; then
    printf "\t\tPerforming linear alignment of SRV to CLS..."
    ./bestlinreg_hires.sh $srv $cls $srv_rsl $cls_rsl $tfm_lin
    printf "\tDone.\n"
fi

if [[ ! -f $srv_lin || "$clobber" == 1 ]]; then
    mincresample -quiet -nearest -clobber -transformation $tfm_lin -like $cls $srv ${srv_lin%.*}
    gzip -f ${srv_lin%.*}
fi
  
###########################
# 3D Non-linear Alignment #
###########################
if [[ "$itr" -gt 0 ]] ; then
    ./downsample.sh $srv_lin $srv_lin_rsl 0.5 0.5 $clobber
    clobber=1
    if [[ ! -f $tfm_nl || "$clobber" -eq "1" ]] ; then
        printf "\t\tNon-linear alignment\n"
        tfm=${srv_lin%/*}/nl_${config}.xfm
        ./bestnlreg_hires.sh $srv_lin $cls $cls_rsl $tfm $config
        xfmconcat -clob $tfm_lin $tfm $tfm_nl
    fi

    if [[ ! -f $srv_nl || "$clobber" == 1 ]]; then
        printf "\t\tResample non-linear\n"
        mincresample -quiet -nearest -clobber -transformation $tfm_nl -like $cls_rsl $srv ${srv_nl%.*}
        gzip -f ${srv_nl%.*}
    fi
    clobber=0
else 
 cp $srv_lin $srv_nl
fi
