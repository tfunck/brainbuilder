#!/bin/bash
#SBATCH --nodes 8
#SBATCH --mem 16G
#SBATCH --time=3:00:00
#SBATCH --job-name=job_reconstruct
#SBATCH --output=%j.out
#SBATCH --account=rpp-aevans-ab
#SBATCH --mail-user=tffunck@gmail.com
#SBATCH --mail-type=ALL


module load singularity/3.5
brain="MR1"
hemisphere="R"
slab=1
out_dir="~/output/"
in_dir="~/receptor_dwn"

while getopts ":s:u:l:" arg; do
  case $arg in
    s) slab=$OPTARG;;
    b) brain=$OPTARG;;
    m) hemisphere=$OPTARG;;
    i) in_dir=$OPTARG;;
    o) out_dir=$OPTARG;;
  esac
done

singularity exec -B /project/def-aevans/tfunck:/project/def-aevans/tfunck  ~/receptor.simg bash -c "python3 ~/julich-receptor-atlas/launch_reconstruction.py -i $in_dir -o $out_dir -s $slab -b $brain --hemi $hemisphere "
