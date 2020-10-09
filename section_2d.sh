#!/bin/bash
#SBATCH --nodes 2
#SBATCH --mem 12G
#SBATCH --time=0:05:00
#SBATCH --job-name=job_reconstruct
#SBATCH --output=%j.out
#SBATCH --account=rpp-aevans-ab
## S BATCH --mail-user=tffunck@gmail.com
## S BATCH --mail-type=ALL


module load singularity/3.6

#singularity exec -B /project/def-aevans/tfunck:/project/def-aevans/tfunck  ~/receptor.simg bash -c "python3 ~/julich-receptor-atlas/section_2d.py $@  "
singularity exec  receptor.simg bash -c "python3 section_2d.py $1 $2 $3 $4 $5 $6 $7 $8 "
