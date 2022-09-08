import numpy as np
import ants
import utils.ants_nibabel as nib
import SimpleITK as sitk
from utils.utils import safe_imread, points2tfm, read_points

vol0 = np.zeros([50,50,50])
vol1 = np.zeros([50,50,50])

vol0[25:35,25:35,25:35] = 1
vol1[20:35,20:35,20:35] = 1

aff_0 = np.array([[1,0,0,-10], 
                  [0,1,0,-20],
                  [0,0,1,-15],
                  [0,0,0,1]])

aff_1 =np.array([[1,0,0,-150], 
                [0,1,0,-150],
                [0,0,1,-100],
                [0,0,0,1]])

fn0='tmp0.nii.gz'
fn1='tmp1.nii.gz'

nib.Nifti1Image(vol0,aff_0).to_filename(fn0)
nib.Nifti1Image(vol1,aff_1).to_filename(fn1)

reg = ants.registration(ants.image_read(fn0), ants.image_read(fn1), 'Similarity', outprefix='./tmp_')

points_fn='points.txt'

print('Points2tfm')
points2tfm(points_fn, 'manual_tfm.mat', fn0, fn1, transform_type='Affine', invert=True, clobber=True )

print('Volume Registration')
vol_tfm = reg['fwdtransforms'][0]
print( ants.get_center_of_mass(ants.image_read(fn0)) )
print( ants.read_transform(vol_tfm).fixed_parameters)
print( ants.read_transform(vol_tfm).parameters)

print( np.max(ants.apply_transforms(ants.image_read(fn0), ants.image_read(fn1), [vol_tfm] ).numpy()) )
print( np.max(ants.apply_transforms(ants.image_read(fn0), ants.image_read(fn1), ['points.mat'] ).numpy()) )

vol_rsl = ants.apply_transforms(ants.image_read(fn0), ants.image_read(fn1), ['manual_tfm.mat'], whichtoinvert=[False] ).numpy()
print( np.max(vol_rsl))
nib.Nifti1Image(vol_rsl, aff_0).to_filename('tmp_rsl.nii.gz')

exit(0)
### SITK



print('Points Registration')
print(ants.read_transform('manual_tfm.mat').fixed_parameters)
print(ants.read_transform('manual_tfm.mat').parameters)






fixed_points, moving_points, fixed_fn, moving_fn = read_points(points_fn, ndim=3)
print(fixed_points)
aff = sitk.AffineTransform(3) 
ref_img = sitk.ReadImage(fn0) 
fpts = sitk.VectorDouble(fixed_points.flatten()) 
mpts = sitk.VectorDouble(moving_points.flatten() ) 
n = fixed_points.shape[0] 
weight = sitk.VectorDouble([1.0] * n) 
tfm = sitk.LandmarkBasedTransformInitializer(aff, fpts, mpts, weight, ref_img, numberOfControlPoints=n ) 
tfm.WriteTransform('points_sitk.mat')
print('Points SimpleITK')
#print(sitk.ReadTransform('points.mat').GetFixedParameters())
#print(sitk.ReadTransform('points.mat').GetParameters())
print(tfm.GetFixedParameters())
print(tfm.GetParameters())


def resample_images(image, reference, transform, interpolation) :
    default_value = np.float64(sitk.GetArrayViewFromImage(image).min())
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolation) 
    resampler.SetReferenceImage(reference)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(transform)
    resampled = resampler.Execute(image)
    return resampled 

img_rsl = resample_images(sitk.ReadImage(fn1), sitk.ReadImage(fn0), tfm, 1) 
print( np.max(sitk.GetArrayFromImage(img_rsl)) ) 
