
#for ligand in "afdx" 'praz' 'rx82' 'sch2' 'pire' 'mk80' 'epib' 'uk14' 'flum' 'dpmg' 'damp' 'sr95' 'musc' 'dpat' 'ly34' 'oxot' 'cgp5' 'ampa' 'kain'  'keta'; do
for ligand in 'musc' ; do
    #sbatch launch_interpolation.sh -b MR1 -m R -l $ligand
    bash launch_interpolation.sh -b MR1 -m R -l $ligand 
    exit 0
done
