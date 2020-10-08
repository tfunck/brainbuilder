
resolution=${1:-3}

n_chunks=30
chunk_perc=`python3 -c "print(1./${n_chunks})"`

for slab in `seq 1 6`; do
    for chunk in `seq 0 $n_chunks`; do 
        launch_reconstruction.sh -r $resolution 
        sbatch launch_reconstruction.sh -r $resolution -c $chunk -p $chunk_perc 
    done
done
