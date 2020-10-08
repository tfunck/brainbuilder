#!/bin/bash
#SBATCH --nodes 8
#SBATCH --mem 12G
#SBATCH --time=12:00:00
#SBATCH --job-name=job_reconstruct
#SBATCH --output=%j.out
#SBATCH --account=rpp-aevans-ab
#SBATCH --mail-user=tffunck@gmail.com
#SBATCH --mail-type=ALL


module load singularity/3.5
brain="MR1"
hemisphere="R"
slab=1
out_dir="/project/def-aevans/tfunck/"
in_dir="~/receptor_dwn/"
resolutions="3 2 1 0.5 0.25"
chunk="0"
chunk_perc="1"

while getopts ":s:b:m:i:o:r:c:p" arg; do
  case $arg in
    s) slab=$OPTARG;;
    b) brain=$OPTARG;;
    m) hemisphere=$OPTARG;;
    i) in_dir=$OPTARG;;
    o) out_dir=$OPTARG;;
    r) resolutions=$OPTARG;;
    c) chunk=$OPTARG;;
    p) chunk_perc=$OPTARG;;
  esac
done

singularity exec -B /project/def-aevans/tfunck:/project/def-aevans/tfunck  ~/receptor.simg bash -c "python3 ~/julich-receptor-atlas/launch_reconstruction.py -i $in_dir -o $out_dir -s $slab -b $brain --hemi $hemisphere --resolutions $resolutions --chunk-perc $chunk_perc --chunk $chunk  "
