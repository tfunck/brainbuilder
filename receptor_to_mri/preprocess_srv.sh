########################################
# Preprocess super-resolution GM image #
########################################
base_srv=$1
srv=$2
slab=$3
receptor_fn=$4
scale_factors_fn=$5


if [[ ! -f  $srv  || "$clobber" == "1" ]]; then
    printf "\t\tCreating truncated version of super-resolution GM mask..."
    python3 slab_borders.py $receptor_fn $scale_factors_fn $slab $base_srv ${srv%.*}  &>> /tmp/log.txt
    crop_from_mask ${srv%.*} ${srv%.*} ${srv%.*}  &>> /tmp/log.txt
	gzip ${srv%.*}
    printf "\tDone.\n"
fi

#if [[ ! -f $mask.gz || "$clobber" == "1" ]]; then
#    mincresample -nearest -clobber -like $base_srv  -transformation tfm_slab_${i}_to_mni.xfm  output/truncate/mr1_brain_mask_slab_${i}.mnc.gz $mask # /tmp/temp.mnc #
#    gzip -f $mask
#fi
#mask=${mask}.gz

