
brain="MR1"
hemisphere="R"
out_dir="/data/receptor/human/output_3_caps/"
out_dir="/data/receptor/human/output_4_caps4real/"
#out_dir="/data/receptor/human/output_2/"

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
singularity exec -B "/data":"/data" ~/receptor.simg bash -c "python3.8 ~/projects/julich-receptor-atlas/launch_reconstruction.py -o $out_dir -b $brain --hemi $hemisphere --mri-gm /data/receptor/human/mri1_R_gm_bg_wm_srv.nii.gz --ndepths 5 "
