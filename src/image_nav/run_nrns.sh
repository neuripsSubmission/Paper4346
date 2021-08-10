#!/bin/bash
set -e
export MAGNUM_LOG=quiet GLOG_minloglevel=2
python -W ignore run.py \
    --dataset 'gibson' \
    --run_type 'curved' \
    --processed_data_file 'test_medium.json.gz' \
    --validity_prediction \
    --feat_prediction \
    --point_nav \
    --switch_func \
    # --pose_noise \
    # --visualize 

