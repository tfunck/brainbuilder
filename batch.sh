
for slab in "1 2 6"; do
    sbatch launch_reconstruction.sh -s $slab -b MR1 -h R
    #sh launch_reconstruction.sh -s $slab -b MR1 -h R
done

