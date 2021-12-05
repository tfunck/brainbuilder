from ants import apply_transforms, image_read, from_numpy
import nibabel as nib
from sys import argv



def safe_ants_image_read(fn) :
    img = nib.load(fn)

    vtr=[0,1,2][0:img.ndim]
    orig_axis=[3]*img.ndim
    spacing = list(img.affine[vtr,vtr] )
    origin = list(img.affine[ vtr,orig_axis] )

    return from_numpy(img.get_fdata(), origin=origin, spacing=spacing)

from scipy.ndimage.measurements import center_of_mass
def apply_transform(fixed, moving, transform, output, extra_tfm=''):

    fixed_ants = safe_ants_image_read(fixed)
    moving_ants = safe_ants_image_read(moving)

    transformlist = [transform]
    if extra_tfm != '' :
        transformlist.append(extra_tfm)

    vol=apply_transforms(fixed_ants, 
                         moving_ants, 
                         transformlist=transformlist, ndim=2, clobber=True).numpy()

    aff = nib.load(fixed).affine
    nib.Nifti1Image(vol, aff ).to_filename(output)

if __name__ == '__main__':

    fixed=argv[1]
    moving=argv[2]
    transform=argv[3]
    output=argv[4]
    print('Fixed:', fixed )
    print('Moving:', moving)
    print('Transform:', transform)
    print('Output:', output)
    apply_transform(fixed, moving, transform, output)


