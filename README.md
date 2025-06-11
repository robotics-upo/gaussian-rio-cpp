# gaussian-rio-cpp

4D Radar-Inertial Odometry implementation in C++, based on Gaussian modeling and multi-hypothesis scan matching

[\[Preprint (arXiv)\]](http://arxiv.org/abs/2412.13639)

**Under Construction**

## Requirements

- CUDA Toolkit (including nvcc)
- ROS (optional). We target the following versions:
	- [ROS 1 One](https://ros.packages.techfak.net/) (currently needed for `upo_gaussian_tester`)
	- ROS 2 Humble
- [small_gicp](https://github.com/koide3/small_gicp)
- Ceres (system package)
- `gaussian_server.py` from [gaussian-rio](https://github.com/robotics-upo/gaussian-rio)

For evaluating NTU4DRadLM and Snail-Radar trajectories:

- numpy
- [evo](https://github.com/MichaelGrupp/evo)

The generated trajectories used to calculate the metrics in our paper can be found [in our website](https://robotics.upo.es/~famozur/gaussian-rio-output.zip).

## Packages

This repository contains two packages:

- `upo_gaussians`: A library providing the actual implementation of the 4D Radar-Inertial Odometry. It provides multiple scan matching algorithms, including Gaussians, NDT, GICP and VGICP. It can be used with or without ROS.
- `upo_gaussian_tester`: A ROS package providing a `run_offline` tool that allows for running the odometry directly on existing datasets (NTU4DRadLM, Snail-Radar)

In order to use the Gaussian odometry, it is currently necessary to run the `gaussian_server.py` script from our previous implementation, which is in charge of generating Gaussian models from radar point clouds. A native optimized implementation of Gaussian fitting is planned to be added in the future, which will remove this requirement.

## Scripts

This repository contains the following scripts (in the `scripts/` folder):

- `run-ntu4dradlm.sh`: Runs the odometry on four sequences of the [NTU4DRadLM dataset](https://github.com/junzhang2016/NTU4DRadLM).
- `run-snail-radar.sh`: Runs the odometry on five sequences of the [Snail-Radar dataset](https://snail-radar.github.io/).
- `evaluate_ntu4dradlm.py`: Generates evaluation metrics for NTU4DRadLM.
- `evaluate_snailradar.py`: Generates evaluation metrics for Snail-Radar.

Before running the odometry, it is required to adjust the path to the datasets in their respective `.yaml` files. Likewise, it is necessary to adjust the `DATASET_DIR` variable in the evaluation scripts.

## Reference

```
@misc{gaussian4drio,
	author = {Fernando Amodeo and Luis Merino and Fernando Caballero},
	title = {4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching},
	year = {2024},
	eprint = {arXiv:2412.13639},
}
```

## Acknowledgements

This work was partially supported by the following grants: 1) INSERTION PID2021-127648OB-C31, and 2) NORDIC TED2021-132476B-I00 projects, funded by MCIN/AEI/ 10.13039/501100011033 and the "European Union NextGenerationEU / PRTR".
