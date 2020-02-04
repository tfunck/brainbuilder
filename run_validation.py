from receptorInterpolate import receptorInterpolate
from nibabel.processing import resample_from_to, resample_to_output
from sys import argv
from plot_interpolation_validation import plot_validation
from os.path import exists
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_template():
    #Load Template
    template=nib.load("output/MR1/R_slab_1/final/flum_space-mni_500um.nii.gz")
    return template 
clobber=False

### Inputs
slab=int(argv[1])
subslab=int(argv[2])

rec_fn="simulation_validation/equivolume_layers_right_mod.nii.gz"
srv_rsl_fn="srv/mri1_gm_bg_srv.nii.gz"
#cls_fn="srv/mri1_gm_bg_srv.nii.gz"
output_dir="~/simulation_validation"
#output_dir="/project/def-aevans/tfunck/output/MR1/R_slab_"+str(slab)+"/ligand/flum/" # /juelich-receptor-atlas/simulation_validation"
ligand="flum"
slice_info_fn=""
rec_df_fn="output/MR1/R_slab_1/coregistration/autoradiograph_info.csv"
transforms_fn="output/MR1/R_slab_1/coregistration/transforms.json"

### Preprocessing Outputs
rec_crop_fn="simulation_validation/equivolume_layers_right_crop.nii.gz"
cls_fn = srv_rsl_crop_fn = "simulation_validation/mri1_gm_bg_srv_slab-1_crop.nii.gz"

template=None 

#Crop REC
if not exists(rec_crop_fn) or clobber :
    print('Crop', rec_fn)
    if template == None : template = load_template() 
    rec_img = nib.load(rec_fn)
    print(template.shape)
    rec_rsl = resample_from_to(rec_img, template)
    print(rec_rsl.shape)
    del template
    del rec_img
    resample_to_output(rec_rsl,voxel_sizes=[0.2,0.02,0.2], order=0 ).to_filename(rec_crop_fn)

    del rec_rsl

#Crop SRV
if not exists(srv_rsl_crop_fn ) or clobber :
    print('Crop', srv_rsl_fn)
    if template == None : template = load_template() 
    srv_rsl = nib.load(srv_rsl_fn)
    srv_rsl_crop = resample_from_to(srv_rsl, template)
    srv_rsl_crop = resample_to_output(srv_rsl_crop, voxel_sizes=[0.2,0.02,0.2], order=1) 
    srv_rsl_crop.to_filename(srv_rsl_crop_fn)

output_fn = output_dir+'/equi_volume_mod_'+str(slab)+'_'+str(subslab)+'validation.nii.gz'

if not exists(output_fn) or clobber : 
    print('run receptorInterpolate')
    
    receptorInterpolate( slab, output_fn, rec_crop_fn, srv_rsl_crop_fn, cls_fn, output_dir, ligand, rec_df_fn,subslab=subslab, clobber=False)


#if not exists("./simulation_validation/flum_validation.csv") or clobber : 
#    exit(1)
#    receptorInterpolate( slab, rec_crop_fn, srv_rsl_crop_fn, cls_fn, output_dir, ligand, rec_df_fn, transforms_fn, clobber=False, validation=True)

gm_img = nib.load(cls_fn)
gm = gm_img.get_data()

flum_img = nib.load(output_fn) #"./simulation_validation/flum.nii.gz")
flum_vol = flum_img.get_data()

ground_truth_img = nib.load(rec_crop_fn)
ground_truth_vol = ground_truth_img.get_data()
errorVolume = np.zeros(flum_vol.shape)
simulation_error=[]
i_list=[]
error_list=[]
for y in range(flum_vol.shape[1]) :
    f = flum_vol[:,y,:]
    g = gm[:,y,:]
    t = ground_truth_vol[:,y,:]
    
    i = (g > 0.1) & (t > 0.1) 
    I = np.sum(i)
    e=np.zeros(t.shape)
    if  I  > 0 and np.sum(f) > 0 :
        e[i] = f[i] / t[i]
        errorVolume[:,y,:] = e
        #e = f / t
        #print(np.sum(f[i]), np.sum(t[i]))
        error = np.mean(e[i])
        #plt.subplot(4,1,1); plt.imshow(g);
        #plt.subplot(4,1,2); 
        #plt.imshow(f);
        #plt.subplot(4,1,3); plt.imshow(t);
        #plt.subplot(4,1,4); plt.imshow(e);
        #plt.title(error)
        #plt.show()
        i_list.append(i)
        error_list.append(error)
    else : 
        error = 0.
    
    simulation_error += [ error ]

nib.Nifti1Image(errorVolume, flum_img.affine).to_filename("simulation_validation/error.nii.gz")
#df = pd.read_csv("simulation_validation/flum_validation.csv" )

plt.plot(simulation_error)
plt.scatter(i_list, error_list, c='r')
plt.show()


