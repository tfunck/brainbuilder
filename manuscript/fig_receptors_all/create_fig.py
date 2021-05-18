
import matplotlib as mpl
COLOR = 'white'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from glob import glob
from scipy.ndimage import center_of_mass
from PIL import Image
import imageio

def crop(ar) :

    s0 = np.sum(ar, axis=0) > 0
    s1 = np.sum(ar, axis=1) > 0
    ar= ar[s1, :]
    return ar[:,s0]

def extract_slices(fn):
    vol = nib.load(fn).get_fdata()
    #vol_ones = np.ones(vol.shape)
    #x,y,z = np.rint( center_of_mass(vol_ones) ).astype(int)
    x,y,z = np.rint(np.array(vol.shape) * 0.5).astype(int)
    print(x,y,z)
    x_slice = np.rot90(vol[x,:,:])
    y_slice = np.rot90(vol[:,y,:])
    z_slice = np.rot90(vol[:,:,z])
    vmax = np.percentile( vol[vol > 0 ], [99.5])[0]
    return x_slice, y_slice, z_slice, vol.min(), vmax

receptors=( ('Glutamate',{'ampa':'AMPA', 'kain':'Kainate', 'mk80':'NMDA', 'ly34':'mGluR2/3'}), 
                ('GABA',{'flum':'GABA$_A$ Benz.', 'cgp5':'GABA$_B$', 'musc':'GABA$_A$ Agonist', 'sr95':'GABA$_A$ Antagonist'}), 
                ('Acetylcholine', {'pire':r'Muscarinic M$_1$', 'afdx':r'Muscarinic M$_2$ (antagonist)','damp':r'Muscarinic M$_3$','epib':r'Nicotinic $\alpha_4\beta_2$','oxot':r'Muscarinic M$_2$ (oxot)'}),
                ('Noradrenalin', {'praz':r'$\alpha_1$','uk14':r'$\alpha_2$ (agonist)','rx82':r'$\alpha_2$ (antagonist)'}),
                ('Serotonin', {'dpat':r'5-HT$_{1A}$','keta':r'5HT$_2$'}),
                ('Dopamine', {'sch2':r"D$_1$"}),
                ('Adenosine', {'dpmg':'Adenosine 1'})
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
        x, y, z, vmin, vmax = extract_slices(fn)
        slices=[x]
        for ii, img in enumerate(slices) :
            #plt.cla()
            #plt.clf()
            #if ii == np.floor(len(slices)/2) : plt.title(receptor,color='white',size=10)
            plt.subplot(4, 5, i+ii+1)
            plt.title(receptor,fontsize=8)
            plt.imshow(img,vmin=vmin, vmax=vmax, origin='upper', cmap='nipy_spectral' )
            plt.axis('off')
            #plt.colorbar()

            #plt.tight_layout()
            #plt.savefig(f'{ligand}.png',facecolor='black')

        i += len(slices);

plt.tight_layout()
plt.savefig('fig_receptors_all.png',facecolor='black')

