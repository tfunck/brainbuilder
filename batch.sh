export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=24

for slab in 1 2 3 4 5 6 ; do
    sh launch_reconstruction.sh -s $slab -b MR1 -m R
done
exit 0
