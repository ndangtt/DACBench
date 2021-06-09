SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
for conf in *.sh
do
    exp_dir=$(echo $conf | cut -d'.' -f1)
    echo "Plotting $exp_dir"
    $SCRIPT_DIR/plot.py $exp_dir
done
