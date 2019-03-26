###############################################
# Preprocess classified receptor volume (CLS) #
###############################################
# Inputs :
cls_ii=$1
clobber=$3
# Outputs
cls_oo=$2

if [[ ! -f $cls_oo || "$clobber" == "1" ]]; then
    printf "\t\tMorph. Closing CLS volume..." 
    mincmorph -clobber -successive DDDEEEE -kernel 3x3x3_y.kern $cls_ii ${cls_oo%.*}  #/tmp/morph.mnc
    #mincblur -quiet -clobber -3dfwhm 0 1 0 -dimensions 2 /tmp/morph.mnc /tmp/morph
    #mincmath -quiet -clobber  -gt -const 0.6 /tmp/morph_blur.mnc ${cls_oo%.*}
    gzip ${cls_oo%.*}
    printf "\tDone.\n"
fi

