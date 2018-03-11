
for mr in MR1 ; do
	for h in R ; do 
  		for slab in slab_1 ; do
			#Step 1 : Apply line removal
			python3 detectLines.py --step 0.2 --train-output model_results/  --raw-source raw/${h}_$slab --raw-output ${mr}/${h}_${slab}/lines_removed 
			#Step 2: Automatic Cropping
			python3 receptorCrop.py --source ${mr}/${h}_${slab}/lines_removed/final --output  ${mr}/${h}_${slab}/crop --step 0.2 --ext ".png"
			
			#Step 3: Coregistration
			python3 receptorRegister.py --slice-order ${mr}_${h}_${slab}_section_numbers.csv --source ${mr}/${h}_${slab}/crop --output ${mr}/${h}_${slab}/coregistered/ --ext "png"
			#Step 4: Reconstruct slices to volume

		done
  	done
done
