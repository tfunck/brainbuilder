
for i in `seq 1 6`; do
  grep  "slab_${i}" $1 > temp.txt
  tag_fn="slab_${i}.tag"
  
  rec_vol=`head -n 1 temp.txt | awk '{split($0, ar, ","); print ar[1];}'`
  mri_vol=`head -n 1 temp.txt | awk '{split($0, ar, ","); print ar[5];}'`

  echo "MNI Tag Point File"  > $tag_fn
  echo "Volumes = 2;"  >> $tag_fn
  echo "%Volume: $rec_vol" >> $tag_fn
  echo "%Volume: $mri_vol" >> $tag_fn
  echo "" >> $tag_fn
  echo "Points =" >> $tag_fn

  while IFS='' read -r line || [[ -n "$line" ]]; do
		rec_vol=`echo $line | awk '{split($0, ar, ","); print ar[1];}'`
		rec_x=`echo $line | awk '{split($0, ar, ","); print ar[2];}'`
		rec_y=`echo $line | awk '{split($0, ar, ","); print ar[3];}'`
		rec_z=`echo $line | awk '{split($0, ar, ","); print ar[4];}'`
		rec_zmax=`mincinfo -dimlength zspace  $rec_vol`
		rec_ymax=`mincinfo -dimlength yspace  $rec_vol`

		echo $rec_z $rec_y $rec_x
		rec_y=`echo $rec_ymax - $rec_y  | bc`	
		rec_z=`echo $rec_zmax - $rec_z  | bc`	
		
		mri_vol=`echo $line | awk '{split($0, ar, ","); print ar[5];}'`
		mri_x=`echo $line | awk '{split($0, ar, ","); print ar[6];}'`
		mri_y=`echo $line | awk '{split($0, ar, ","); print ar[7];}'`
		mri_z=`echo $line | awk '{split($0, ar, ","); print ar[8];}'`
		
		mri_zmax=`mincinfo -dimlength zspace $mri_vol`
		mri_ymax=`mincinfo -dimlength yspace $mri_vol`	
		
		mri_y=`python3 -c "print( $mri_ymax - $mri_y )"`
		mri_z=`python3 -c "print( $mri_zmax - $mri_z )"`	
			
		rec_w=`voxeltoworld $rec_vol $rec_z $rec_y $rec_x`
		mri_w=`voxeltoworld $mri_vol $mri_z $mri_y $mri_x`

		echo $rec_w $mri_w \"\" >> $tag_fn
  done < temp.txt
  echo "$(cat $tag_fn);" > $tag_fn
done
