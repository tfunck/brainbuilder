from sys import argv
from apply_transforms import safe_ants_image_read
import nibabel as nib

if __name__ == '__main__'  :
    fn = argv[1]
    out_fn = argv[2]
    vol = safe_ants_image_read(fn)
    nib.Nifti1Image(vol.numpy(), nib.load(fn).affine).to_filename(out_fn)
