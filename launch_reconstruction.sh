#!/bin/bash
#SBATCH --nodes 8
#SBATCH --mem 12G
#SBATCH --time=0:0:01
#SBATCH --job-name=job_reconstruct
#SBATCH --output=%j.out
#SBATCH --account=rpp-aevans-ab
echo SBATCH --mail-user=tffunck@gmail.com
echo SBATCH --mail-type=ALL


module load singularity/3.6
brain="MR1"
hemisphere="R"
slab=1
out_dir="/project/def-aevans/tfunck/output/"
in_dir="~/receptor_dwn/"
resolution="3"

while getopts "s:b:m:i:o:r:c:p:" arg; do
  case $arg in
    s) slab=$OPTARG;;
    b) brain=$OPTARG;;
    m) hemisphere=$OPTARG;;
    i) in_dir=$OPTARG;;
    o) out_dir=$OPTARG;;
    r) resolution=$OPTARG;;
  esac
done

singularity exec -B /project/def-aevans/tfunck:/project/def-aevans/tfunck  ~/receptor.simg bash -c "python3 -c \"import numpy; print(\"okay\")\""
#singularity exec -B /project/def-aevans/tfunck:/project/def-aevans/tfunck  ~/receptor.simg bash -c "python3 ~/julich-receptor-atlas/launch_reconstruction.py  -i $in_dir -o $out_dir -s $slab -b $brain --hemi $hemisphere --resolution $resolution   "
