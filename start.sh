conda activate dacbench

# dacbench & searl
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$DIR/:$PYTHONPATH
export PYTHONPATH=$DIR/solve_onell/searl/SEARL/:$PYTHONPATH

# pfrl
export PYTHONPATH=$DIR/../pfrl:$PYTHONPATH
