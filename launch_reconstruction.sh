
brain="MR1"
hemisphere="R"
out_dir="/data/receptor/human/output_3_caps/"
out_dir="/data/receptor/human/output_4_caps4real/"
#out_dir="/data/receptor/human/output_5/"

while getopts "s:b:m:i:o:r:c:p:s:" arg; do
  case $arg in
    s) slab=$OPTARG;;
    b) brain=$OPTARG;;
    m) hemisphere=$OPTARG;;
    i) in_dir=$OPTARG;;
    o) out_dir=$OPTARG;;
  esac
done

mkdir -p $out_dir

rm /data/receptor/human/output_4_caps4real/5_surf_interp/surfaces/slab-1_MR1_gray_surface_R_81920.surf*

singularity exec -B "/data":"/data" ~/receptor-v.1.2.simg bash -c "python3.8 ~/projects/julich-receptor-atlas/launch_reconstruction.py -o $out_dir -b $brain --hemi $hemisphere --mri-gm /data/receptor/human/mri1_R_gm_bg_srv.nii.gz --cortex-gm /data/receptor/human/mri1_gm_srv.nii.gz --ndepths 10 --surf-dir /data/receptor/human/civet/mri1/surfaces/ -r 4.0 3.0 2.0 1.0 "


#rm /data/receptor/human/output_4_caps4real/5_surf_interp/surfaces/slab-1_MR1_gray_surface_R_81920.surf*gii
#singularity exec -B "/data":"/data" ~/receptor-v.1.2.simg bash -c "python3.8 ~/projects/julich-receptor-atlas/launch_reconstruction.py -o $out_dir -b $brain --hemi $hemisphere --mri-gm /data/receptor/human/mri1_R_gm_bg_srv.nii.gz --ndepths 5 --surf-dir /data/receptor/human/civet/mri1/surfaces/ "
#/data/receptor/human/civet/mri1/surfaces/

#bash test_ants_transform.sh
