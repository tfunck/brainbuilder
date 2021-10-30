#!/bin/bash
#SBATCH --nodes 4
#SBATCH --mem 18G
#SBATCH --time=04:59:00
#SBATCH --job-name=job_reconstruct
#SBATCH --output=%j.out
#SBATCH --account=rpp-aevans-ab

#module load singularity/3.6
brain="MR1"
hemisphere="R"
out_dir="/data/receptor/human/output_2/"
resolution="3"

while getopts "s:b:m:i:o:r:c:p:s:" arg; do
  case $arg in
    s) slab=$OPTARG;;
    b) brain=$OPTARG;;
    m) hemisphere=$OPTARG;;
    i) in_dir=$OPTARG;;
    o) out_dir=$OPTARG;;
  esac
done

mkdir -p $out_dir
singularity exec -B "/data":"/data" ~/receptor_new/ bash -c "python3.7 ~/projects/julich-receptor-atlas/launch_reconstruction.py -i /home/receptor/human/ -o $out_dir -b $brain --hemi $hemisphere --ndepths 20 "
