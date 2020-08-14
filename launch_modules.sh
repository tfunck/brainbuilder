wait_for_files(){
    file_list=$1
    finished_files=`ls $file_list 2> /dev/null | wc -l`
    total_files=`echo $file_list | wc -w`

    echo $finished_files 
    while [[ $finished_files != $total_files ]]; do
        finished_files=`ls $file_list 2> /dev/null | wc -l`
        sleep 60
    done
}

###
### Parameters
###
brain_list="MR1" # MR2 MR3"
hemi_list="R" #L
slab_list="1 2 3 4 5 6"
ligand_list="flum"

### If running on server:
#run=qsub
### If running locally:
run=sh 

#Path to autoradiograph_info.csv
autoradiograph_info_fn='autoradiograph_info.csv'

out_dir='reconstruction_output'
src_dir='receptor_dwn/'
crop_dir=$out_dir/0_crop
mkdir -p $out_dir

###---------------------###
###  PROCESSING STEPS   ###
###---------------------###
#   0. Crop all autoradiographs
#   1. Init Alignment (Rigid 2D, per slab)
#   2. GM segmentation of receptor volumes (per slab)
#   3. Slab to MRI (Affine 3D, per slab)
#   4. GM MRI to autoradiograph volume (Nonlinear 3D, per slab)
#   5. Autoradiograph to GM MRI (2D nonlinear, per slab)
#   6. Interpolate autoradiograph to surface
#   7. Interpolate missing vertices on sphere, interpolate back to 3D volume

### Step 0 :
python3 reconstruction_crop.py -s $src_dir -o $crop_dir
###
### Steps 1 & 2: Initial interslab alignment, GM segmentation
###
#Paramaters
n_epochs=10

init_file_check=""
for brain in $brain_list ; do
    for hemi in $hemi_list ; do
        for slab in $slab_list; do 
            out_dir_1="${out_dir}/1_init_align/${brain}_${hemi}_${slab}/"
            out_dir_2="${out_dir}/2_segment/${brain}_${hemi}_${slab}/"
            init_align_fn="${out_dir_1}/brain-${brain}_hemi-${hemi}_slab-${slab}_init_align.nii.gz"
            seg_fn=${out_dir_2}"/brain-${brain}_hemi-${hemi}_slab-${slab}_seg.nii.gz"
            
            if [[ ! -f $init_align_fn || ! -f $seg_fn ]]; then
                bash -c "$run batch_init_alignment_and_seg.sh $brain $hemi $slab $init_align_fn $crop_dir $out_dir_1 $autoradiograph_info_fn $n_epochs $out_dir_2 $seg_fn"
            fi
            init_file_check="$init_file_check $init_align_fn $seg_fn "
        done
        break
    done
    break
done

#Pause script until previous job have finished 
#wait_for_files "$init_file_check"

###
### Step 3 : Align slabs to MRI
###
echo Step 3: align slabs to MRI
init_subslab_check=""
for brain in $brain_list ; do
    for hemi in $hemi_list ; do
        bash -c "$run batch_align_slab_to_mri.sh $out_dir ${out_dir}/3_align_slab_to_mri/${brain}_${hemi}/ $brain $hemi" 
        #init_file_check="$init_file_check $init_align_fn $seg_fn "
    done
done

#Pause script until previous job have finished 
#wait_for_files $init_subslab_check

###
### Step 4 : Nonlinear alignment of MRI to slab
###
nonlinear_file_check=""
for brain in $brain_list; do
    for hemi in $hemi_list; do
        for slab in $slab_list; do
            seg_dir="${out_dir}/2_segment/${brain}_${hemi}_${slab}/"
            align_dir="${out_dir}/3_align_slab_to_mri/${brain}_${hemi}/final"
            nl_dir="${out_dir}/4_nonlinear/${brain}_${hemi}_${slab}/"

            #fixed_fn=`ls ${seg_dir}/*_slab-${slab}_seg.nii.gz`
            fixed_fn=`ls ${align_dir}/cls*_${slab}_*.nii.gz`
            moving_fn=`ls ${align_dir}/srv*_${slab}_*.nii.gz`
            init_fn=`ls ${align_dir}/affine_${slab}_*.h5`

            tfm_fn=${nl_dir}/${brain}_${hemi}_${slab}_nl.h5
            nonlinear_file_check="$nonlinear_file_check $tfm_fn"
            echo Fixed: $fixed_fn
            echo Moving: $moving_fn
            echo "Transform: $init_fn"
            if [[ ! -f $tfm_fn ]]; then
                bash -c "$run batch_nonlinear_alignment.sh $brain $hemi $slab $fixed_fn $moving_fn $init_fn $tfm_fn"
            fi
        done
    done
done

#Pause script until previous job have finished 
wait_for_files $nonlinear_file_check 

###
### Step 5 : 2D alignment of receptor to resample MRI GM vol
###
nonlinear_2d_check=""
for brain in $brain_list; do
    for hemi in $hemi_list; do
        for slab in $slab_list; do
            nl_2d_out_dir="${out_dir}/5_nonlinear_2d/${brain}_${hemi}_${slab}/"
            nl_dir="${out_dir}/4_nonlinear/${brain}_${hemi}_${slab}/"

            nonlinear_2d_check="$intraslab_interpolation_check $interpolated_fn"
            rec_fn="reconstruction_output/1_init_align/${brain}_${hemi}_${slab}/brain-${brain}_hemi-${hemi}_slab-${slab}_init_align.nii.gz"
            srv_fn="srv/mri1_gm_bg_srv.nii.gz"
            srv_rsl_fn="${nl_2d_out_dir}/${brain}_${hemi}_${slab}_nl_level-0_GC_SyN.nii.gz"
            df_fn="reconstruction_output/1_init_align/${brain}_${hemi}_${slab}/final/${brain}_${hemi}_${slab}_final.csv"
            lin_tfm_fn=`cat reconstruction_output/3_align_slab_to_mri/MR1_R/final/best_slab_position.csv | grep affine_${slab} | awk '{ split($0,ar,","); print ar[6] }'`
            nl_tfm_fn=${nl_dir}/${brain}_${hemi}_${slab}_nllevel-0_GC_SyN_Composite.h5
            out_fn="${nl_2d_out_dir}/${brain}_${hemi}_${slab}_nl_2d.nii.gz"
            
            direction=`python3 -c "import json; print(json.load(open('scale_factors.json','r'))[\"$brain\"][\"$hemi\"][\"$slab\"][\"direction\"])"`
            if [[ ! -f $interpolated_fn ]]; then
                bash -c "$run batch_2d_nonlinear_alignment.sh $df_fn $rec_fn $srv_fn $srv_rsl_fn $lin_tfm_fn $nl_tfm_fn $nl_2d_out_dir $out_fn"
            fi
        done
    done
done

#Pause script until previous job have finished 
wait_for_files $intraslab_interpolation_check

###
### Step 6 : Interpolate missing receptor densities using cortical surface mesh
###
interslab_interpolation_check=""
for brain in $brain_list; do
    for hemi in $hemi_list; do
        for ligand in $ligand_list ; do 
            out_fn=""
            interslab_interpolation_check="$interslab_interpolation_check $interpolated_fn"
            bash -c "$run batch_interpolate.sh  "
        done
    done
done
