

source=$1
target=$2
source_lo=$3
target_lo=$4
final_tfm=$5
output_img=$6

m="-xcorr"

mincblur -clobber -fwhm 12 $source_lo /tmp/src_1
mincblur -clobber -fwhm 12 $target_lo /tmp/tgt_1

minctracc -clobber -est_translation $m -source_mask $source -simplex 12 -step 3 3 3 -lsq3 /tmp/src_1_blur.mnc /tmp/tgt_1_blur.mnc /tmp/tfm_1_1.xfm 

mincblur -clobber -fwhm 8 $source_lo /tmp/src_1
mincblur -clobber -fwhm 8 $target_lo /tmp/tgt_1
minctracc -clobber -transformation /tmp/tfm_1_1.xfm -step 4 4 4 -simplex 4 $m -lsq6  /tmp/src_1_blur.mnc /tmp/tgt_1_blur.mnc /tmp/tfm_1_2.xfm 

mincblur -clobber -fwhm 4 $source_lo /tmp/src_1
mincblur -clobber -fwhm 4 $target_lo /tmp/tgt_1
minctracc -clobber -transformation /tmp/tfm_1_2.xfm -step 2 2 2 -simplex 4 $m -lsq9  /tmp/src_1_blur.mnc /tmp/tgt_1_blur.mnc /tmp/tfm_1_3.xfm

mincblur -clobber -fwhm 2 $source_lo /tmp/src_1
mincblur -clobber -fwhm 2 $target_lo /tmp/tgt_1
minctracc -clobber -transformation /tmp/tfm_1_3.xfm -step 1 1 1 -simplex 2 $m -lsq12  /tmp/src_1_blur.mnc /tmp/tgt_1_blur.mnc /tmp/tfm_1_4.xfm

mincblur -clobber -fwhm 0.5 $source /tmp/src_1
mincblur -clobber -fwhm 0.5 $target /tmp/tgt_1
minctracc -clobber -transformation /tmp/tfm_1_4.xfm -step 1 1 1 -simplex 2 $m -lsq12  /tmp/src_1_blur.mnc /tmp/tgt_1_blur.mnc /tmp/tfm_1_5.xfm

cp /tmp/tfm_1_5.xfm $final_tfm

#mincresample -nearest -clobber -transformation /tmp/tfm_1_3.xfm -tfm_input_sampling $source /tmp/temp.mnc
#register /tmp/temp.mnc $output_img
#exit 0

