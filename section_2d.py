from sys import argv
from utils.utils import shell
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tempfile

def section_2d( prefix, mv_rsl_fn, fx_fn,  out_hires_fn, s_str, f_str, lin_itr_str, nl_itr_str) :

    stout, sterr, errorcode = shell(f'antsRegistration -v 0 -d 2 --write-composite-transform 1  --initial-moving-transform [{fx_fn},{mv_rsl_fn},1] -o [{prefix}_,{out_hires_fn},/tmp/out_inv.nii.gz] -t Similarity[.1] -c {lin_itr_str}  -m Mattes[{fx_fn},{mv_rsl_fn},1,20,Regular,1] -s {s_str} -f {f_str}   -t Affine[.1]   -c {lin_itr_str}  -m Mattes[{fx_fn},{mv_rsl_fn},1,20,Regular,1] -s {s_str} -f {f_str} -t SyN[0.1] -m Mattes[{fx_fn},{mv_rsl_fn},1,20,Regular,1] -c [{nl_itr_str}] -s {s_str} -f {f_str}', exit_on_failure=True,verbose=False)

    #Create QC image
    #out_2d_np = nib.load(out_hires_fn).get_fdata()
    #plt.imshow(srv_np,cmap='gray',origin='lower')
    #plt.imshow(out_2d_np,cmap='hot',alpha=0.45,origin='lower')
    #plt.tight_layout()
    #plt.savefig(f'{prefix}qc.png')
    #plt.clf()
    #plt.cla()

if __name__ == '__main__' :
    section_2d(*argv[1:])
