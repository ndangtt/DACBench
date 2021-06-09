config=$1
rm -rf $config; python -W ignore ../SEARL/scripts/run_searl_td3.py --expt_dir $config --config $config.yml
