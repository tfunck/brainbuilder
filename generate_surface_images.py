from utils.mesh_utils import volume_to_surface, load_mesh_ext
from matplotlib_surface_plotting import plot_surf
import numpy as np
import os


def generate_average_surf_plot(wm_fn,gm_fn,volume_fn,out_fn,delta=0.1):

    if os.path.exists(wm_fn) and os.path.exists(gm_fn) and os.path.exists(volume_fn) :

        wm_coords, wm_faces = load_mesh_ext(wm_fn, correct_offset=True)
        gm_coords, _ = load_mesh_ext(gm_fn, correct_offset=True)

        d = wm_coords - gm_coords

        values = np.zeros(gm_coords.shape[0])
        n = np.zeros(gm_coords.shape[0])

        for depth in np.arange(0,1.0,delta):
            print('depth',depth)
            mid_coords = gm_coords + depth * d

            
            temp_values, idx = volume_to_surface(mid_coords, volume_fn)
            
            values[idx] += temp_values
            n[idx] += 1

        values[n>0] /= n[n>0]

        vmin, vmax = np.percentile(values,[2,98])
        print('Writing', out_fn)
        plot_surf(mid_coords, wm_faces, values, rotate=[90,270], filename=out_fn,
              vmax = vmax, vmin = vmin,
              pvals=np.ones_like(values), cmap_label='fmol/mg protein') 


if __name__ == '__main__' :

    wm_fn = '/data/receptor/human/civet/mri1/surfaces/MR1_white_surface_R_81920.surf.gii'
    gm_fn = '/data/receptor/human/civet/mri1/surfaces/MR1_gray_surface_R_81920.surf.gii'

    volume_fn=f'/data/receptor/human/output_5/5_surf_interp/MR1_R_flum_0.25mm_l25_space-mni.nii.gz'
    out_fn='/data/receptor/human/output_5/5_surf_interp/MR1_R_flum_0.25mm_l25_space-mni_surf.png'
    
    generate_average_surf_plot(wm_fn,gm_fn,volume_fn,out_fn,delta=0.1)

    wm_fn = 'macaque/rh_11530_mf/surfaces/11530_white_surface_L_0.white'
    gm_fn = 'macaque/rh_11530_mf/surfaces/11530_gray_surface_L_0.pial'

    volume_fn=f'macaque/output_ms/rh_11530_mf/ligand/11530_L_flum_1mm_l10_space-mni.nii.gz'
    out_fn=f'macaque/output_ms/rh_11530_mf/ligand/11530_L_flum_1mm_l10_space-mni_surf.png'
    
    generate_average_surf_plot(wm_fn,gm_fn,volume_fn,out_fn,delta=0.1)
