import imageio
from skimage.transform import resize
from glob import glob
import os
import numpy as np
import utils.ants_nibabel as nib


aff=np.eye(4)
aff[0:3,0:3] *= 0.01
print(aff);

fn_list = glob('rat/L6/img_orig.bkp/*tif')
print(fn_list)

for fn in fn_list:
    out_fn='rat/L6/img_orig/'+os.path.splitext(os.path.basename(fn))[0] +'.nii.gz'
    if not os.path.exists(out_fn) or True:
        img=imageio.imread(fn)
        #img=resize(img, np.rint(np.array(img.shape)/10), order=5)
        print('1.', fn)
        img = np.max(img) - img
        #img = np.rot90(img,-1)
        #img = np.fliplr(img)
        img = np.flipud(img.T)
        nib.Nifti1Image(img,aff).to_filename(out_fn)
        print('2.', out_fn)
