out_dir="macaque/output_ohbm2023/"

singularity exec   ~/projects/julich-receptor-atlas/receptor.simg bash -c "python3.8 ~/projects/julich-receptor-atlas/launch_macaque.py rh_11530_mf macaque/rh_11530_mf/ templates/MEBRAINS_segmentation_NEW_gm_bg_left.nii.gz macaque/rh_11530_mf_points_lpi.txt macaque/rh_11530_mf/scale_factors.json ${out_dir}" 

#singularity exec -B "/data":"/data" ~/receptor.simg bash -c "python3.8 ~/projects/julich-receptor-atlas/launch_reconstruction.py -o $out_dir -b $brain --hemi $hemisphere --ndepths 30 "
