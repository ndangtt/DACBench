conf=$1
logFn=$conf/log.txt
python ../train.py --exp_dir $conf --config $conf.yml >$logFn 2>&1 & tail -f $logFn
