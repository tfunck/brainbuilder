src=$1
tgt=$2
tfm=$3

weights="-w_translations 1 1 0 -w_rotations 0.0 0.0 0.0174533 -w_scales 0.02 0.02 0.0 -w_shear  0.20 0.0 0.0"
qdir="/home/t/neuro/projects/Juelich-Receptor-Atlas/receptor_to_mri/quarantines/Linux-x86_64/"
source ${qdir}init.sh
step0="-step 2 2 0 "
step1="-step 1 1 0 "
step2="-step 1 1 0 "
step3="-step 0.5 0.5 0 "
s0="4 4 0"
s1="2 2 0"
s2="1 1 0"
#s3="0.5 0.5 0"

tfm0=/tmp/tfm0.xfm
tfm1=/tmp/tfm1.xfm
tfm2=/tmp/tfm2.xfm
tfm3=/tmp/tfm3.xfm
base="-clobber $weights -source_mask $src -model_mask $tgt"

# 0
mincblur -quiet -clobber  -3dfwhm $s0 -dimensions 2 $src /tmp/src
mincblur -quiet -clobber  -3dfwhm $s0 -dimensions 2 $tgt /tmp/tgt
minctracc  $base -identity $step0 -simplex 4 -lsq12  /tmp/src_blur.mnc /tmp/tgt_blur.mnc $tfm0

# 1
mincblur -quiet -clobber  -3dfwhm $s1 -dimensions 2 $src /tmp/src
mincblur -quiet -clobber  -3dfwhm $s1 -dimensions 2 $tgt /tmp/tgt
minctracc  $base  -transformation $tfm0 -identity $step1 -simplex 2 -lsq12  /tmp/src_blur.mnc /tmp/tgt_blur.mnc $tfm1

# 2
mincblur -quiet -clobber  -3dfwhm $s2 -dimensions 2 $src /tmp/src
mincblur -quiet -clobber  -3dfwhm $s2 -dimensions 2 $tgt /tmp/tgt
minctracc  $base -transformation $tfm1 $step2 -simplex 1 -lsq12  /tmp/src_blur.mnc /tmp/tgt_blur.mnc $tfm2

cp $tfm2 $tfm
#3
#mincblur -quiet -clobber  -3dfwhm $s3 -dimensions 2 $src /tmp/src
#mincblur -quiet -clobber  -3dfwhm $s3 -dimensions 2 $tgt /tmp/tgt
#minctracc  $base -transformation $tfm1 $step3 -simplex 0.5 -lsq12  /tmp/src_blur.mnc /tmp/tgt_blur.mnc $tfm3

