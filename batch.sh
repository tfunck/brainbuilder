
resolution=${1:-3}

for slab in `seq 1 6`; do
    sbatch launch_reconstruction.sh -s $slab -r $resolution 
    exit 0
done
