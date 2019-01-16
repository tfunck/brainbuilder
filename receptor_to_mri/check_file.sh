fn=$1

if [[ -f $fn ]]; then
    echo $fn
else 
    echo "Error: could not find $fn"
    exit 1
fi
