from utils.mesh_utils import display_surface, volume_to_surface, load_mesh_ext
from matplotlib_surface_plotting import plot_surf
import numpy as np


def generate_average_surf_plot(wm_fn,gm_fn,volume_fn,out_fn,delta=0.1):

    delta = 0.1

    wm_coords, wm_faces = load_mesh_ext(wm_fn)
    gm_coords, _ = load_mesh_ext(gm_fn)

    d = wm_coords - gm_coords

    values = np.zeros(gm_coords.shape[0])

    for depth in np.arange(0,1.0,delta):
        print('depth',depth)
        mid_coords = gm_coords + depth * d

        values += delta * volume_to_surface(mid_coords, volume_fn)

    vmin, vmax = np.percentile(values,[2,98])

    plot_surf(mid_coords, wm_faces, values, rotate=[90,270], filename=out_fn,
          vmax = vmax, vmin = vmin,
          pvals=np.ones_like(values), cmap_label='fmol/mg protein') 




if __name__ == '__main__' :

    wm_fn = '/data/receptor/human/civet/mri1/surfaces/MR1_white_surface_R_81920.surf.gii'
    gm_fn = '/data/receptor/human/civet/mri1/surfaces/MR1_gray_surface_R_81920.surf.gii'

    volume_fn=f'/data/receptor/human/output_5/5_surf_interp/MR1_R_{flum}_0.25mm_l25_space-mni.nii.gz'
    out_fn='/data/receptor/human/output_5/5_surf_interp/MR1_R_flum_0.25mm_l25_space-mni_surf.png'

    generate_average_surf_plot(wm_fn,gm_fn,volume_fn,out_fn,delta=0.1)

