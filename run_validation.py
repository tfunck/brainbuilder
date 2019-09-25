from receptorInterpolate import receptorInterpolate
from nibabel.processing import resample_from_to, resample_to_output

from plot_interpolation_validation import plot_validation
from os.path import exists
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

clobber=False
### Inputs
slab=1
rec_fn="simulation_validation/equivolume_layers_right.mnc"
srv_rsl_fn="srv/mri1_gm_bg_srv.nii.gz"
#cls_fn="srv/mri1_gm_bg_srv.nii.gz"
output_dir="./simulation_validation"
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
    if template == None : template=nib.load("output/MR1/R_slab_1/final/flum_space-mni_500um.nii.gz")
    print('img_rsl')
    img_rsl = resample_from_to(nib.load(rec_fn), template, order=1)
    print('resample_to_output')
    resample_to_output(img_rsl,voxel_sizes=[0.2,0.02,0.2], order=1 ).to_filename(rec_crop_fn)
    print('done')

#Crop SRV
if not exists(srv_rsl_crop_fn ) or clobber :
    if template == None : template = load_template() 
    srv_rsl = nib.load(srv_rsl_fn)
    srv_rsl_crop = resample_from_to(srv_rsl, template)
    srv_rsl_crop = resample_to_output(srv_rsl_crop, voxel_sizes=[0.2,0.02,0.2], order=1) 
    srv_rsl_crop.to_filename(srv_rsl_crop_fn)

clobber=True
if not exists("./simulation_validation/flum.nii.gz") or clobber : 
    print('run receptorInterpolate')
    receptorInterpolate( slab, rec_crop_fn, output_dir+'/flum_validation.nii.gz', srv_rsl_crop_fn, cls_fn, output_dir, ligand, rec_df_fn, clobber=False)


#if not exists("./simulation_validation/flum_validation.csv") or clobber : 
#    exit(1)
#    receptorInterpolate( slab, rec_crop_fn, srv_rsl_crop_fn, cls_fn, output_dir, ligand, rec_df_fn, transforms_fn, clobber=False, validation=True)

gm_img = nib.load(cls_fn)
gm = gm_img.get_data()

flum_img = nib.load("./simulation_validation/flum.nii.gz")
flum_vol = flum_img.get_data()

ground_truth_img = nib.load(rec_crop_fn)
ground_truth_vol = ground_truth_img.get_data()
errorVolume = np.zeros(flum_vol.shape)
simulation_error=[]
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
    else : 
        error = 0.
    
    simulation_error += [ error ]

nib.Nifti1Image(errorVolume, flum_img.affine).to_filename("simulation_validation/error.nii.gz")
df = pd.read_csv("simulation_validation/flum_validation.csv" )

plt.plot(simulation_error)
plt.scatter(df["i"].values, df["error"].values, c='r')
plt.show()


