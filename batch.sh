
for slab in "6"; do
    sbatch launch_reconstruction.sh -s $slab -b MR1 -h R
    #bash launch_reconstruction.sh -s $slab -b MR1 -h R
done

