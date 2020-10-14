from sys import argv
from utils.utils import shell
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tempfile

def section_2d( prefix, mv_rsl_fn, fx_fn,  out_hires_fn, s_str, f_str, lin_itr_str, nl_itr_str, batch_fn) :
    command_str = f'antsRegistration -v 0 -d 2 --write-composite-transform 1  --initial-moving-transform [{fx_fn},{mv_rsl_fn},1] -o [{prefix}_,{out_hires_fn},/tmp/out_inv.nii.gz] -t Similarity[.1] -c {lin_itr_str}  -m Mattes[{fx_fn},{mv_rsl_fn},1,20,Regular,1] -s {s_str} -f {f_str}   -t Affine[.1]   -c {lin_itr_str}  -m Mattes[{fx_fn},{mv_rsl_fn},1,20,Regular,1] -s {s_str} -f {f_str} -t SyN[0.1] -m Mattes[{fx_fn},{mv_rsl_fn},1,20,Regular,1] -c [{nl_itr_str}] -s {s_str} -f {f_str}'
    if batch_fn == None :
        stout, sterr, errorcode = shell(command_str, exit_on_failure=True,verbose=False)
    else :
        print('\t\t\tWriting',batch_fn)
        F = open(batch_fn, 'w+')
        F.write("#!/bin/bash\n")
        F.write("#SBATCH --nodes 4\n")
        F.write("#SBATCH --mem 6G\n")
        F.write("#SBATCH --time=00:03:00\n")
        F.write("#SBATCH --job-name=job_reconstruct\n")
        F.write("#SBATCH --output=%j.out\n")
        F.write("#SBATCH --account=rpp-aevans-ab\n")
        F.write("module load singularity/3.6\n")
        F.write(f"singularity exec -B /project/def-aevans/tfunck:/project/def-aevans/tfunck  ~/receptor.simg {command_str}\n")


if __name__ == '__main__' :
    section_2d(*argv[1:])
