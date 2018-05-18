
python3 detectLines.py --epochs 5 --step 0.2 --train-source "test/" --train-output "line_detection_model/" 

for mr in MR1 ; do
	for h in R ; do 
  		for slab in slab_2 slab_3 slab_4 slab_5 slab_6 ; do
		  	echo 
		    echo $slab
			echo
			#Step 1 : Apply line removal
			
			python3 detectLines.py --step 0.2 --train-source "test/" --train-output line_detection_model/  --raw-source raw/${h}_$slab --raw-output ${mr}/${h}_${slab}/lines_removed
			#python3 detectLines.py --step 0.2   --raw-source raw/${h}_$slab --raw-output ${mr}/${h}_${slab}/lines_removed
			
			#Step 2: Automatic Cropping
			echo python3 receptorCrop.py --source ${mr}/${h}_${slab}/lines_removed/final --output  ${mr}/${h}_${slab}/crop --step 0.2 --ext ".png"
			python3 receptorCrop.py --source ${mr}/${h}_${slab}/lines_removed/ --output  ${mr}/${h}_${slab}/crop --step 0.2 --ext ".png"
			
			#Step 3: Coregistration
			#python3 receptorRegister.py --slice-order ${mr}_${h}_${slab}_section_numbers.csv --source ${mr}/${h}_${slab}/crop --output ${mr}/${h}_${slab}/coregistered/ --ext "png" --tiers  "flum,musc,afdx,uk18,sr95,cgp5,pire;sch2,rx82,keta,dpat,ly34;kain,damp,mk80;oxot;epib" #--clobber  #--start-tier 3  


			#Step 4:
			#python3 receptorRegisterNL.py  --source  ${mr}/${h}_${slab}/coregistered/ --output ${mr}/${h}_${slab}/nl_aligned --ext "png"  --clobber  #--start-tier 3  


			#Step 5: Reconstruct slices to volume
			#python3 slices2mnc.py --source ${mr}/${h}_${slab}/coregistered/ --output-dir ${mr}/${h}_${slab}/volume   --output-file ${mr}_${h}_${slab}.mnc --receptor-csv  ${mr}/${h}_${slab}/coregistered/receptor_slices.csv
			#"flum,musc,afdx,uk19,sr95,cgp5,pire;sch2,rx82,keta,oxot,dpat,ly34;kain,damp,mk80;epib"

		done
  	done
done
