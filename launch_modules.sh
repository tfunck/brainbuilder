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
slab_list="1" # 2 3 4 5 6"
ligand_list="flum"

#If running on server:
#run=qsub
#If running locally:
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
            init_align_fn="${out_dir_1}/brain-${brain}_hemi-${hemi}_slab-${slab}_init_align_${n_epochs}.nii.gz"
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
exit 0

#Pause script until previous job have finished 
wait_for_files "$init_file_check"

###
### Step 3 : Align slabs to MRI
###
echo Step 3: align slabs to MRI
init_subslab_check=""
for brain in $brain_list ; do
    for hemi in $hemi_list ; do
        bash -c "$run batch_align_slab_to_mri.sh $out_dir ${out_dir}/3_align_slab_to_mri/${brain}_${hemi}/ $brain $hemi" 
        init_file_check="$init_file_check $init_align_fn $seg_fn "
    done
    exit 0   
done

#Pause script until previous job have finished 
wait_for_files $init_subslab_check

###
### Step 4 : Nonlinear alignment of MRI to slab
###
nonlinear_file_check=""
for brain in $brain_list; do
    for hemi in $hemi_list; do
        for slab in $slab_list; do
            tfm_fn=${out_dir}/'init_align_brain-${brain}_hemi-${hemi}_slab-${slab}.nii.gz'
            nonlinear_file_check="$nonlinear_file_check $tfm_fn"

            if [[ ! -f $tfm_fn ]]; then
                qbatch batch_mri_to_receptor.sh $tfm_fn
            fi
        done
    done
done

#Pause script until previous job have finished 
wait_for_files $nonlinear_file_check 

###
### Step 5 : Within slab interpolation
###
intraslab_interpolation_check=""
for brain in $brain_list; do
    for hemi in $hemi_list; do
        for slab in $slab_list; do
            interpolated_fn=""
            intraslab_interpolation_check="$intraslab_interpolation_check $interpolated_fn"
            if [[ ! -f $interpolated_fn ]]; then
                qbatch batch_interpolate.sh 
            fi
        done
    done
done

#Pause script until previous job have finished 
wait_for_files $intraslab_interpolation_check

###
### Step 6 : Between slab interpolation
###
interslab_interpolation_check=""
for brain in $brain_list; do
    for ligand in $ligand_list ; do 
        out_fn=""
        interslab_interpolation_check="$interslab_interpolation_check $interpolated_fn"
        qbatch batch_interpolate.sh 
    done
done
