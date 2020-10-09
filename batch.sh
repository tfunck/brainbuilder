
resolution=${1:-3}

n_chunks=30
chunk_perc=`python3 -c "print(1./${n_chunks})"`

echo $resolution
echo $chunk_perc
for slab in `seq 2 6`; do
    for chunk in `seq 0 $n_chunks`; do 
        sbatch launch_reconstruction.sh -s $slab -p $chunk_perc -r $resolution -c $chunk  
        exit 0
    done
done
