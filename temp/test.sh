
original_fx_fn="QW#HG#MR1s1#R#damp#5671#17#L#L.nii.gz"
original_mv_fn="QW#HG#MR1s1#R#epib#5657#17#L#L.nii.gz"

echo check manual tfm

manual_tfm='MR1_R_1_epib_559.0_points_affine.mat' 
points='MR1_R_1_epib_559.0_points.txt'
affine='affine.h5'


rm test1.nii.gz $affine

echo $points
cat $points
echo
echo
echo

python3 -c "from utils import points2tfm; points2tfm('${points}','$affine', transform_type='Rigid', ndim=2, clobber=True)"

python3 apply_transforms.py $original_fx_fn $original_mv_fn $affine test1.nii.gz 561_final_Rigid.h5

antsApplyTransforms -v 1 -d 2 -i vol_mv.nii.gz -r vol_fx.nii.gz -t ${affine} -o test2.nii.gz


register test1.nii.gz test2.nii.gz #$original_fx_fn

#antsApplyTransforms -v 1 -d 2 -i $original_32_fn -r $original_33_fn -t $manual_tfm -t 33_final_Rigid.h5 -o test2.nii.gz
#antsApplyTransforms -v 1 -d 2 -i $original_33_fn -r $original_33_fn -t 33_final_Rigid.h5 -o test3.nii.gz

#niihd test1.nii.gz $original_32_fn $original_33_fn
#register $original_33_fn $original_32_fn
#register test1.nii.gz $original_33_fn




