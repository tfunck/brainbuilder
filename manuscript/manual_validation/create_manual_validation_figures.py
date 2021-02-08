import nibabel as nib
import pandas as pd
import os
import numpy as np
from subprocess import call, Popen, PIPE, STDOUT

newlines = ['\n', '\r\n', '\r']
def unbuffered(proc, stream='stdout'):
    stream = getattr(proc, stream)
    with contextlib.closing(stream):
        while True:
            out = []
            last = stream.read(1)
            # Don't loop forever
            if last == '' and proc.poll() is not None:
                break
            while last not in newlines:
                # Don't loop forever
                if last == '' and proc.poll() is not None:
                    break
                out.append(last)
                last = stream.read(1)
            out = ''.join(out)
            print(out)
            yield out

def shell(cmd, verbose=False,exit_on_failure=True):
    '''Run command in shell and read STDOUT, STDERR and the error code'''
    stdout=""
    if verbose :
        print(cmd)


    process=Popen( cmd, shell=True, stdout=PIPE, stderr=STDOUT, universal_newlines=True, )

    for line in unbuffered(process):
        stdout = stdout + line + "\n"
        if verbose :
            print(line)

    errorcode = process.returncode
    stderr=stdout
    if errorcode != 0:
        print ("Error:")
        print ("Command:", cmd)
        print ("Stderr:", stdout)
        if exit_on_failure : exit(errorcode)
    return stdout, stderr, errorcode

def tfm_coords(aligned_fn, resolution, tfm_fn, coords_fn, coords_out_fn) :
    aligned_img = nib.load(aligned_fn)
    x_start, y_start, z_start = aligned_img.affine[3,0:3]
    
    x_range = [x_start + resolution * x for x in range(aligned_img.shape[0])]
    y = y_start + slab_order * 0.02
    z_range = [z_start + resolution * z for z in range(aligned_img.shape[2])]
    
    coords=[]
    for x in x_range :
        for z in z_range :
           coords.append([x,y,z]) 
    np.savetxt( coords_fn,np.array(coords))
    
    shell(f'antsApplyTransformsToPoints -d 3 -i {coords_fn} -t [tfm_fn,1]  -o {coords_out_fn}',verbose=True)

if __name__ == '__main__':
    
    img_list=("RD#HG#MR1#3#R#oxot#5774#17","RD#HG#MR1#3#R#oxot#5774#18","RE#hg#MR1#4#R#oxot#5863#05", "RG#hg#MR1#6#R#oxot#5689#13" )
    data_dir="/scratch/tfunck/output2/"
    resolution=0.5
    df_fn="../../section_numbers/autoradiograph_info.csv"
    df = pd.read_csv(df_fn)

    for img_base in img_list :
        row = df.loc[ df["base"] == img_base ]

        slab = row["slab"].values[0]
        slab_order = row["slab"].values[0]
        ligand = row["ligand"].values[0]
        
        init_fn=f"{data_dir}/MR1_R_{slab}/1_init_align//brain-MR1_hemi-R_slab-{slab}_init_align.nii.gz"
        aligned_fn=f"{data_dir}/MR1_R_{slab}/{resolution}mm/4_nonlinear_2d/MR1_R_{slab}_nl_2d.nii.gz"
        tfm_fn =  f"{data_dir}/MR1_R_{slab}/{resolution}mm/3_align_slab_to_mri/rec_to_mri_Composite.h5"
        ligand_fn=f"{data_dir}/MR1_R_{slab}/{resolution}mm/5_surf_interp/MR1_R_{ligand}_0.5mm.nii.gz"

        coords_fn = f'coords_{img_base}'
        coords_out_fn = f'coords_{img_base}'

        if not os.path.exists(coords_out_fn) :
            tfm_coords(aligned_fn, resolution, tfm_fn, coords_fn, coords_out_fn)
        
        xdim, ydim, zdim = nib.load(aligned_fn).shape

        coords_rsl = np.loadtxt(coords_out_fn)

        ligand_img = nib.load(ligand_fn)
        x_start, y_start, z_start = ligand_img.affine[3,0:3]
        print(x_start, y_start, z_start)
        ligand_vol = ligand_img.get_fdata()
        coords_idx = np.rint((coords_rsl - np.array([xstart, ystart,zstart])) / np.array([resolution, 0.02,resolution ]))
        ar = ligand_vol[coords_idx].reshape(xdim,zdim) 

        print(ar.shape)


