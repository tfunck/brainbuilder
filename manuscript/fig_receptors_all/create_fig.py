import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from glob import glob
from scipy.ndimage import center_of_mass
from PIL import Image

def crop(ar) :

    s0 = np.sum(ar, axis=0) > 0
    s1 = np.sum(ar, axis=1) > 0
    ar= ar[s1, :]
    return ar[:,s0]

def extract_slices(fn):
    vol = nib.load(fn).get_fdata()
    x,y,z = np.rint( center_of_mass(vol) ).astype(int)
    x_slice = np.rot90(vol[x,:,:])
    y_slice = np.rot90(vol[:,y,:])
    z_slice = np.rot90(vol[:,:,z])
    return x_slice, y_slice, z_slice

receptors=( ('Glutamate',{'ampa':'AMPA'}), 
                ('GABA',{'flum':'GABA$_A$ Benz.', 'cgp5':'GABA$_B$', 'musc':'GABA$_A$ Agonist', 'sr95':'GABA$_A$ Antagonist'}), 
                ('Acetylcholine', {'afdx':r'Muscarinic M$_2$ (antagonist)','damp':r'Muscarinic M$_3$'}),
                ('Noradrenalin', {}),
                ('Serotonin', {}),
                ('Dopamine', {}),
                ('Adrenalin', {}),
                ('Adenosine', {})
                )

file_dir = argv[1]
i=0

plt.figure(figsize=(6, 8), dpi=600, facecolor='b', edgecolor='b')

for family, family_dict in receptors :
    
    for ligand, receptor in family_dict.items() :
        fn_list =glob(f'{file_dir}/*{ligand}*nii.gz') 
        if len(fn_list) == 0 : continue
        else : fn = fn_list[0]
        slices = extract_slices(fn)
        i += 3;

        for ii, img in enumerate(slices) :
            plt.subplot(2, 3*3, i+ii+1)
            if ii == 1 : plt.title(receptor,color='white',size=4)
            plt.imshow(img,origin='upper', cmap='nipy_spectral' )
            plt.axis('off')

plt.tight_layout()
plt.savefig('fig_receptors_all.png',facecolor='black')

