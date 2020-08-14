brain=$1
hemi=$2
slab=$3
fixed_fn=$4
moving_fn=$5
init_tfm=$6
out_fn=$7
python3 reconstruction_nonlinear_alignment.py -b $brain --hemi $hemi -s $slab -f $fixed_fn -m $moving_fn -o $out_fn -t $init_tfm   
