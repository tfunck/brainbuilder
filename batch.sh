
for slab in `seq 6 6`; do
    sbatch launch_reconstruction.sh -s $slab
done

