
#for slab in 1 2 3 4 5 6 ; do
for slab in 1 ; do
    sbatch launch_reconstruction.sh -s $slab -b MR1 -m R
    #bash launch_reconstruction.sh -s $slab -b MR1 -m R
done
exit 0
