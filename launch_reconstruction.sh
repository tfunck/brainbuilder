#!/bin/bash
#SBATCH --nodes 4
#SBATCH --mem 10G
#SBATCH --time=02:30:00
#SBATCH --job-name=job_reconstruct
#SBATCH --output=%j.out
#SBATCH --account=rpp-aevans-ab

module load singularity/3.6
brain="MR1"
hemisphere="R"
slab=1
out_dir="/project/def-aevans/tfunck/output/"
in_dir="~/receptor_dwn/"
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

singularity exec -B /project/def-aevans/tfunck:/project/def-aevans/tfunck  ~/receptor.simg bash -c "python3 ~/julich-receptor-atlas/launch_reconstruction.py --remote -i $in_dir  --mri-gm ~/srv/mri1_gm_bg_srv.nii.gz  -o $out_dir -s $slab -b $brain --hemi $hemisphere   "
