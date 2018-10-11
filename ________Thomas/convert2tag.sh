n_slabs=3
base_mri="/home/t/neuro/projects/julich/Juelich-Receptor-Atlas/mri/mr1_mri_T1_rsl.mnc"
################
# Fix tag file #
################
sed -e ':loop' -e 's/;;/;/g' -e 't loop' $1 | sed 's/;/,/g' | sed 's/nii/mnc/g' | sed 's/.gz//g' > temp0.txt


##############
# 
#############

#for i in `seq 1 $n_slabs`; do
	#out_fn="mni_to_virtual_slab_${i}.xfm"
	#out_invert_fn="mni_to_virtual_slab_${i}_invert.xfm"
	#xform_fn="mr1_mri_T1_rsl_rotate_for_slab_${i}L.xform"
	#echo "MNI Transform File" > $out_fn
	#echo "% Created from tag file mri" >> $out_fn
	#echo "% using 6 parameter linear transformation with Papaya" >> $out_fn
	#echo "" >> $out_fn
	#echo "Transform_Type = Linear;" >> $out_fn
	#echo "Linear_Transform =" >> $out_fn
	#cat $xform_fn >> $out_fn
	
#done

###########
#
############
for i in `seq 1 $n_slabs`; do
  echo hello 0
  grep  "slab_${i}" temp0.txt > temp.txt
  tag_fn="slab_${i}.tag"
  xfm_tag_fn="tag_lin_${i}.xfm"
  xfm_rec_to_mri_fn="mri_lin_${i}.xfm"
  rec_vol=`head -n 1 temp.txt | awk '{split($0, ar, ","); print ar[1];}'`
  mri_vol=`head -n 1 temp.txt | awk '{split($0, ar, ","); print ar[5];}'`

  xfm_mri_to_virtual_slab="mni_to_virtual_slab_${i}.xfm"
  xfm_virtual_slab_to_mri="virtual_slab_to_mni_${i}.xfm"
  if [[ ! -f $xfm_mri_to_virtual_slab ]] ; then
  	minctracc -lsq6 -est_translation -clobber $base_mri $mri_vol $xfm_mri_to_virtual_slab
  	xfminvert -clobber $xfm_mri_to_virtual_slab $xfm_virtual_slab_to_mri
  fi

  echo "MNI Tag Point File"  > $tag_fn
  echo "Volumes = 2;"  >> $tag_fn
  echo "%Volume: $rec_vol" >> $tag_fn
  echo "%Volume: $mri_vol" >> $tag_fn
  echo "" >> $tag_fn
  echo "Points =" >> $tag_fn
  cat $tag_fn
  while IFS='' read -r line || [[ -n "$line" ]]; do
		rec_vol=`echo $line | awk '{split($0, ar, ","); print ar[1];}'`
		rec_x=`echo $line | awk '{split($0, ar, ","); print ar[2];}'`
		rec_y=`echo $line | awk '{split($0, ar, ","); print ar[3];}'`
		rec_z=`echo $line | awk '{split($0, ar, ","); print ar[4];}'`
		
		rec_zmax=`mincinfo -dimlength zspace  $rec_vol`
		rec_ymax=`mincinfo -dimlength yspace  $rec_vol`

		rec_y=`echo $rec_ymax - $rec_y  | bc`	
		rec_z=`echo $rec_zmax - $rec_z  | bc`	
	
		mri_vol=`echo $line | awk '{split($0, ar, ","); print ar[5];}'`
		mri_x=`echo $line | awk '{split($0, ar, ","); print ar[6];}'`
		mri_y=`echo $line | awk '{split($0, ar, ","); print ar[7];}'`
		mri_z=`echo $line | awk '{split($0, ar, ","); print ar[8];}'`
		
		#echo $mri_vol $mri_x $mri_y $mri_z	
			
		mri_zmax=`mincinfo -dimlength zspace $mri_vol`
		mri_ymax=`mincinfo -dimlength yspace $mri_vol`	

		mri_y=`python3 -c "print( $mri_ymax - $mri_y )"`
		mri_z=`python3 -c "print( $mri_zmax - $mri_z )"`	
			
		rec_w=`voxeltoworld $rec_vol $rec_z $rec_y $rec_x`
		mri_w=`voxeltoworld $mri_vol $mri_z $mri_y $mri_x`

		echo $rec_w $mri_w \"\" >> $tag_fn
		#echo "rec" $rec_x $rec_y $rec_z "mri" $mri_x $mri_y $mri_z
		echo "rec", $rec_w, "mri", $mri_w
  		rec_rsl_lin=`echo $rec_vol | sed 's/.mnc/_rsl_lin.mnc/'`
  		rec_rsl_vtl=`echo $rec_vol | sed 's/.mnc/_virtual_slab.mnc/'`
  done < temp.txt
  echo "$(cat $tag_fn);" > $tag_fn


  echo $tag_fn $xfm_tag_fn
  tagtoxfm -clobber -lsq12 $tag_fn $xfm_tag_fn
  xfmconcat -clobber $xfm_mri_to_virtual_slab $xfm_tag_fn  $xfm_rec_to_mri_fn
  #xfmconcat -clobber  $xfm_tag_fn $xfm_virtual_slab_to_mri  $xfm_rec_to_mri_fn

  mincresample -clobber -tricubic -invert_transformation -transformation $xfm_rec_to_mri_fn -like $base_mri $rec_vol $rec_rsl_lin
  register $base_mri $rec_rsl_lin
  sum_list="$rec_rsl_lin $sum_list"
done


minccalc -clobber -expr "sum(A)" $sum_list receptor_volume_affine.mnc
