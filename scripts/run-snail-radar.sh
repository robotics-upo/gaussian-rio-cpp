#!/usr/bin/env bash

mkdir -p output/snail-radar

rosrun upo_gaussian_tester run_offline --output ./output/snail-radar/eagle_ndt --config rio-ndt.yaml --dataset snail-radar.yaml --radar oculii_eagle --imu mti3dk
rosrun upo_gaussian_tester run_offline --output ./output/snail-radar/eagle_gicp --config rio-gicp.yaml --dataset snail-radar.yaml --radar oculii_eagle --imu mti3dk
rosrun upo_gaussian_tester run_offline --output ./output/snail-radar/eagle_vgicp --config rio-vgicp.yaml --dataset snail-radar.yaml --radar oculii_eagle --imu mti3dk
FORCE_NUM_PARTICLES=1 rosrun upo_gaussian_tester run_offline --output ./output/snail-radar/eagle_gaussian_p1 --config rio-gaussian-eagle.yaml --dataset snail-radar.yaml --radar oculii_eagle --imu mti3dk
FORCE_NUM_PARTICLES=8 rosrun upo_gaussian_tester run_offline --output ./output/snail-radar/eagle_gaussian_p8 --config rio-gaussian-eagle.yaml --dataset snail-radar.yaml --radar oculii_eagle --imu mti3dk

rosrun upo_gaussian_tester run_offline --output ./output/snail-radar/ars_gicp --config rio-gicp.yaml --dataset snail-radar.yaml --radar ars548 --imu mti3dk
rosrun upo_gaussian_tester run_offline --output ./output/snail-radar/ars_vgicp --config rio-vgicp.yaml --dataset snail-radar.yaml --radar ars548 --imu mti3dk
FORCE_NUM_PARTICLES=1 rosrun upo_gaussian_tester run_offline --output ./output/snail-radar/ars_gaussian_p1 --config rio-gaussian-ars.yaml --dataset snail-radar.yaml --radar ars548 --imu mti3dk
FORCE_NUM_PARTICLES=8 rosrun upo_gaussian_tester run_offline --output ./output/snail-radar/ars_gaussian_p8 --config rio-gaussian-ars.yaml --dataset snail-radar.yaml --radar ars548 --imu mti3dk
