import nibabel as nib 
import ants 
import numpy as np 
from scipy.ndimage.measurements import center_of_mass 

from utils import points2tfm

def r(x): return np.round(x,2) 

def check(fn, tfm_fn): 
    img = nib.load(fn) 
    vol = img.get_fdata() 
    com = center_of_mass(vol) 
    aff = img.affine 
    com1 = ants.get_center_of_mass( ants.image_read(fn) ) 
    #print("scipy:\t\t", r([com[1] * aff[0,0]+aff[0,3], com[0]*aff[1,1]+aff[1,3]]) ) 
    #print("scipy (no origin)\t\t",r([com[1] * aff[0,0], com[0]*aff[1,1]]) ) 
    print("antspy:\t\t", r(com1)) 
    print('antsReg:\t\t',r(ants.read_transform(tfm_fn).fixed_parameters)  ) 
    print() 


#check('588_Rigid-0/tmp_level-0_Mattes_Rigid.nii.gz','588_Rigid-0/_level-0_Mattes_Rigid_Composite.h5') 
#check('588_Rigid-0/_level-0_Mattes_Rigid.nii.gz','588_Rigid-0/level-0_Mattes_Rigid_Composite_Concatenated.h5') 
#check('633_Rigid-0/tmp_level-0_Mattes_Rigid.nii.gz','633_Rigid-0/_level-0_Mattes_Rigid_Composite.h5')
#check('633_Rigid-0/_level-0_Mattes_Rigid.nii.gz','633_Rigid-0/level-0_Mattes_Rigid_Composite_Concatenated.h5')




original_fx_fn="QW#HG#MR1s1#R#damp#5671#17#L#L.nii.gz"
original_mv_fn="QW#HG#MR1s1#R#epib#5657#17#L#L.nii.gz"

points2tfm('MR1_R_1_epib_559.0_points.txt','test_affine.h5', transform_type='Rigid', ndim=2, clobber=True, reference=original_mv_fn)
points2tfm('MR1_R_1_epib_559.0_points.txt','test_affine_no_offset.h5', transform_type='Rigid', ndim=2, clobber=True )

from utils import shell


shell(f'antsApplyTransforms -v 1 -d 2 -i {original_fx_fn} -r {original_mv_fn} -t test_affine_no_offset.h5 -o test_manual_no_offset.nii.gz')
shell(f'antsApplyTransforms -v 1 -d 2 -i {original_fx_fn} -r {original_mv_fn} -t test_affine.h5 -o test_manual.nii.gz')

shell(f'register {original_fx_fn} test_manual.nii.gz test_manual_no_offset.nii.gz')

