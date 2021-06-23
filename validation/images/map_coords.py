import h5py 
import nibabel as nib
import matplotlib.pyplot as plt

nl2d_fn = 'MR1_R_3_nl_2d_0.4mm.nii.gz'
surf_fn_list=['slab-3_surf_0.4mm_0.96_rsl.h5']


def get_section_intervals(vol):

    valid_sections = np.sum(vol, axis=(0,2)) > 0
    plt.plot(valid_sections)
    plt.show()


img = nib.load(nl2d_fn)
vol = img.get_fdata()

get_section_intervals(vol)

