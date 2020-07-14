pause(){
    file_list=$1
    finished_files=0
    total_files=`echo $file_list | wc -w`
    while [[ $finished_files != $total_files ]]; do
        finished_files=`ls $file_list 2> /dev/null | wc -l`
        sleep 300
    done
}

###
### Parameters
###
brain_list="MR1 MR2 MR3"
hemi_list="R L"
slab_list="1 2 3 4 5 6"
ligand_list="flum"

###
### Step 1 : Initial interslab alignment
###
init_file_check=""
for brain in brain_list; do
    for hemi in hemi_list; do
        for slab in "1" "2" "3" "4" "5" "6"; do
            init_align_fn=${out_dir}/'init_align_brain-${brain}_hemi-${hemi}_slab-${slab}.nii.gz'
            seg_fn=${out_dir}/'seg_brain-${brain}_hemi-${hemi}_slab-${slab}.nii.gz'
            
            if [[ ! -f $init_align_fn && ! -f $seg_fn ]]; then
                echo qsub batch_interslab_alignment.sh $brain $hemi $slab    
                init_file_check="$init_file_check $init_align_fn $seg_fn  "
            fi
        done
    done
done

#Pause script until previous job have finished 
pause $init_file_check 

###
### Step 2 : Align slabs to MRI
###
init_subslab_check=""
qsub batch_find_subslab.sh 

#Pause script until previous job have finished 
pause $init_subslab_check

###
### Step 3 : Nonlinear alignment of MRI to slab
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
pause $nonlinear_file_check 

###
### Step 4 : Within slab interpolation
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
pause $intraslab_interpolation_check

###
### Step 5 : Between slab interpolation
###
interslab_interpolation_check=""
for brain in $brain_list; do
    for ligand in $ligand_list ; do 
        out_fn=""
        interslab_interpolation_check="$interslab_interpolation_check $interpolated_fn"
        qbatch batch_interpolate.sh 
    done
done
