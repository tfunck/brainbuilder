brain=$1
hemi=$2
slab=$3
fixed_fn=$4
moving_fn=$5
init_tfm=$6
out_fn=$7
shrink_factors=$8
smooth_factors=$9
iterations=${10}
echo hello
echo brain $brain
echo hemi $hemi
echo slab $slab
echo fixed $fixed
echo moving $moving_fn
echo init $init_tfm
echo out $out_fn
echo shrink $shrink_factors
echo smooth  $smooth_factors
echo iterations $iterations
echo python3 reconstruction_nonlinear_alignment.py -b $brain --hemi $hemi -s $slab -f $fixed_fn -m $moving_fn -o $out_fn -t $init_tfm  --shrink-factors $shrink_factors --smooth-factors $smooth_factors --iterations $iterations
exit 0

