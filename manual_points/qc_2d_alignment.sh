#!/bin/bash
mkdir -p temp_dice_images

base_output_dir="/data/receptor/human/output_2"

rsync -Praz $JULICH:${base_output_dir}/MR*_*_*/1_init_align/dice//dice*csv .

for dice_csv in `ls dice*csv`; do 
    while IFS=',' read -r i brain hemisphere slab ligand moving fixed dice y fixed_y  moving_orig fixed_orig ; do
        if [[  "$ligand" != "ligand" ]]; then
        
            output_dir="${base_output_dir}/${brain}_${hemisphere}_${slab}/"
            dice_dir="${output_dir}/1_init_align/dice"

            pass_fn="2d/${brain}_${hemisphere}_${slab}_${ligand}_${y}.pass"

            if [[ ! -f $pass_fn ]]; then
                
                points_fn="2d/${brain}_${hemisphere}_${slab}_${ligand}_${y}_points.txt"
                echo $moving
                qc_fn="${dice_dir}/dice-${dice}_y-${y}_ligand-$ligand.png"
                local_qc_fn=temp_dice_images/dice-${dice}_y-${y}_ligand-$ligand.png


                rsync -Praz $JULICH:$moving_orig temp_dice_images/
                rsync -Praz $JULICH:$fixed_orig temp_dice_images/
                rsync -Praz $JULICH:$qc_fn temp_dice_images/
                
                eog $local_qc_fn

                echo "Pass (y)?"
                read -p "Pass (y)? " pass_flag  

                if [[ $pass_flag != "y" ]]; then
                    echo "Warning!"
                    echo "Save points as t"
                    echo "Script will automatically move t to $points_fn"
                    register temp_dice_images/`basename ${fixed_orig}` temp_dice_images/`basename ${moving_orig}`
                    echo mv t.tag $points_fn
                    mv t.tag $points_fn
                fi 

                if [[ -f $points_fn ]] ; then
                    echo $i,$dice,$y,$ligand,$slab,$moving_orig > $pass_fn
                fi

            fi
        fi
    done < $dice_csv
done
