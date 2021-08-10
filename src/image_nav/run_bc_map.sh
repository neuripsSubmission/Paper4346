#!/bin/bash
set -e
export MAGNUM_LOG=quiet GLOG_minloglevel=2
BASEDIR="/nethome/mhahn30/Repositories/fair_internship/image_nav"
python -W ignore run.py \
    --behavioral_cloning \
    --bc_type 'map'\
    --dataset 'mp3d' \
    --run_type 'curved' \
    --processed_data_file 'test_easy.json.gz' \

