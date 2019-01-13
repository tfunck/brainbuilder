iterations=0

./preprocess.sh

for i in `seq 0 $iterations`; do
    echo "Iteration: $i"
    # Align SRV to CLS
    #./3d_alignment.sh $i 0

    # Refine 2D slice-wise alignment
    ./2d_alignment.sh

    exit 1
done

# Ligand
./ligand.sh

