
for slab in "6"; do
    sbatch launch_reconstruction.sh -s $slab -b MR1 -m R
    #bash launch_reconstruction.sh -s $slab -b MR1 -m R
done
exit 0
