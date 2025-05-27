#!/usr/bin/env bash

mkdir -p output/ntu4dradlm

rosrun upo_gaussian_tester run_offline --output ./output/ntu4dradlm/ndt --config rio-ndt.yaml --dataset ntu4dradlm.yaml --radar oculii_eagle --imu vectornav
rosrun upo_gaussian_tester run_offline --output ./output/ntu4dradlm/gicp --config rio-gicp.yaml --dataset ntu4dradlm.yaml --radar oculii_eagle --imu vectornav
rosrun upo_gaussian_tester run_offline --output ./output/ntu4dradlm/vgicp --config rio-vgicp.yaml --dataset ntu4dradlm.yaml --radar oculii_eagle --imu vectornav
FORCE_NUM_PARTICLES=1 rosrun upo_gaussian_tester run_offline --output ./output/ntu4dradlm/gaussian_p1 --config rio-gaussian-eagle.yaml --dataset ntu4dradlm.yaml --radar oculii_eagle --imu vectornav
FORCE_NUM_PARTICLES=8 rosrun upo_gaussian_tester run_offline --output ./output/ntu4dradlm/gaussian_p8 --config rio-gaussian-eagle.yaml --dataset ntu4dradlm.yaml --radar oculii_eagle --imu vectornav
