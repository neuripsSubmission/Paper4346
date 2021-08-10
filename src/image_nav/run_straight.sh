#!/bin/bash
set -e
export MAGNUM_LOG=quiet GLOG_minloglevel=2
python -W ignore run.py \
    --dataset 'mp3d' \
    --run_type 'straight' \
    --processed_data_file 'test_hard.json.gz' \
    --feat_prediction \
    --point_nav \
    --switch_func \
    --visualize \

    # --validity_prediction \


    # --pose_noise \S
    # --actuation_noise 
    # --visualize \
    # --single \
    # --behavioral_cloning \


    # --dataset 'gibson' \
    # --run_type 'straight' \
    # --processed_data_file 'test_easy2.json.gz' \
    # --behavioral_cloning
    # --bc_type "map"