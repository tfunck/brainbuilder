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

receptors=( ('Glutamate',{'ampa':'AMPA', 'kain':'Kainate', 'mk80':'NMDA', 'ly34':'mGluR2/3'}), 
                ('GABA',{'flum':'GABA$_A$ Benz.', 'cgp5':'GABA$_B$', 'musc':'GABA$_A$ Agonist', 'sr95':'GABA$_A$ Antagonist'}), 
                ('Acetylcholine', {'pire':r'Muscarinic M$_1$', 'afdx':r'Muscarinic M$_2$ (antagonist)','damp':r'Muscarinic M$_2$','epib':r'Nicotinic $\alpha_4\beta_2$','oxot':r'Muscarinic M$_3$ (oxot)'}),
                ('Noradrenalin', {'praz':r'$\alpha_1$','uk14':r'$\alpha_2$ (agonist)','rx82':r'$\alpha_2$ (antagonist)'}),
                ('Serotonin', {'dpat':r'5-HT$_{1A}$','keta':r'5HT$_2$'}),
                ('Dopamine', {'sch2':r"D$_2$"}),
                ('Adenosine', {'dpmg':'Adenosine 2'})
                )

file_dir = argv[1]
i=0

plt.figure(figsize=(8, 6), dpi=600, facecolor='b', edgecolor='b')

for family, family_dict in receptors :
    
    for ligand, receptor in family_dict.items() :
        fn_list =glob(f'{file_dir}/*{ligand}*nii.gz') 
        if len(fn_list) == 0 : 
            print('skiping ligand', ligand)
            continue
        else : fn = fn_list[0]
        slices = [extract_slices(fn)[0]]
        print(i, fn)
        for ii, img in enumerate(slices) :
            plt.subplot(4, 5, i+ii+1)
            
            if ii == np.floor(len(slices)/2) : plt.title(receptor,color='white',size=10)
            plt.imshow(img,origin='upper', cmap='nipy_spectral' )
            plt.axis('off')
        i += len(slices);

plt.tight_layout()
plt.savefig('fig_receptors_all.png',facecolor='black')

