out_dir="/data/receptor/macaque/output_ohbm2023/"

singularity exec  -B "/data":"/data" ~/receptor.simg bash -c "python3.8 ~/projects/julich-receptor-atlas/launch_macaque.py rh_11530_mf /data/receptor/macaque/rh_11530_mf/ /data/receptor/macaque/templates/MEBRAINS_segmentation_NEW_gm_bg_left.nii.gz /data/receptor/macaque/rh_11530_mf/scale_factors.json ${out_dir}" 

#singularity exec   ~/projects/julich-receptor-atlas/receptor.simg bash -c "python3.8 ~/projects/julich-receptor-atlas/launch_macaque.py rh_11530_mf /data/receptor/macaque/rh_11530_mf/ /data/receptor/macaque/templates/MEBRAINS_segmentation_NEW_gm_bg_left.nii.gz /data/receptor/macaque/rh_11530_mf/scale_factors.json ${out_dir}" 

