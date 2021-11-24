mkdir -p temp_dice_images

while IFS=',' read -r i brain hemisphere slab moving fixed dice y ligand moving_orig fixed_orig ; do
    if [[  "$ligand" != "ligand" ]]; then
    
        pass_fn="2d/${brain}_${hemisphere}_${slab}_${ligand}_${y}.pass"

        if [[ ! -f $pass_fn ]]; then
            
            points_fn="2d/${brain}_${hemisphere}_${slab}_${ligand}_${y}_points.txt"
            qc_fn="dice-${dice}_y-${y}_ligand-$ligand.png"


            rsync -Praz $JULICH:$moving_orig temp_dice_images/
            rsync -Praz $JULICH:$fixed_orig temp_dice_images/
            rsync -Praz $JULICH:$qc_fn temp_dice_images/
            
            eog $qc_fn

            echo "Pass (y)?"
            read -p "Pass (y)? " pass_flag  

            if [[ $pass_flag != "y" ]]; then
                echo "Warning!"
                echo "Save points as temp.txt"
                echo "Script will automatically move temp.txt to $points_fn"
                register images/`basename ${fixed_orig}` images/`basename ${moving_orig}`
                echo mv temp.txt $points_fn
                mv temp.txt $points_fn
            fi 

            echo $i,$dice,$y,$ligand,$slab,$moving_orig > $pass_fn

        fi
    fi
done < $1
