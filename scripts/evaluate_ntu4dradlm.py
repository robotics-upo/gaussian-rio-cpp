import numpy as np

from evo.core import sync
from evo.core.units import Unit
from evo.core.trajectory import PoseTrajectory3D
from evo.core.metrics import RPE, PoseRelation, StatisticsType
from evo.tools import file_interface

from typing import Tuple, List

DATASET_DIR = '/mnt/data/work/datasets/radar'

GT_PATTERN   = DATASET_DIR + '/{seq}/gt_odom.txt'
PRED_PATTERN = './output/ntu4dradlm/{method}_traj_{seq}.txt'
METHODS      = ['ekfrio', 'ndt', 'gicp', 'vgicp', 'gaussian_p1', 'gaussian_p8']
SEQUENCES    = ['cp', 'nyl', 'loop2', 'loop3']

def get_path(pattern, method, seq):
	x = pattern
	x = x.replace('{method}', method)
	x = x.replace('{seq}', seq)
	return x

def truncate(num, places):
	x = 10**places
	return int(num*x) / x

def mean_pose_period(traj:PoseTrajectory3D) -> float:
	return (traj.timestamps[-1] - traj.timestamps[0]) / traj.num_poses

def rpe_metric(gt, pred, which, delta):
	m = RPE(pose_relation=which, delta=delta, delta_unit=Unit.meters, all_pairs=True)
	m.process_data((gt, pred))
	return m.get_statistic(StatisticsType.mean) / delta

def evaluate(traj_gt:PoseTrajectory3D, traj_pred:PoseTrajectory3D) -> Tuple[float,float]:
	gt_raw_len = traj_gt.path_length

	diff_gt = mean_pose_period(traj_gt)
	diff_pred = mean_pose_period(traj_pred)

	traj_gt, traj_pred = sync.associate_trajectories(traj_gt, traj_pred, max_diff=max(diff_gt,diff_pred))

	accum_tran = []
	accum_rot = []

	for distperc in [ 0.1, 0.2, 0.3, 0.4, 0.5 ]:
		delta = truncate(distperc*gt_raw_len, 2)

		try:
			m_tran = rpe_metric(traj_gt, traj_pred, PoseRelation.translation_part, delta)
			m_rot  = rpe_metric(traj_gt, traj_pred, PoseRelation.rotation_angle_deg, delta)
		except:
			continue

		accum_tran.append(m_tran)
		accum_rot.append(m_rot)

	accum_tran = np.mean(accum_tran)
	accum_rot  = np.mean(accum_rot)

	return float(accum_tran), float(accum_rot)

for method in METHODS:
	for seq in SEQUENCES:
		print('Evaluating', method, seq)
		traj_gt = file_interface.read_tum_trajectory_file(get_path(GT_PATTERN,method,seq))
		traj_pred = file_interface.read_tum_trajectory_file(get_path(PRED_PATTERN,method,seq))
		rpe_tran, rpe_rot = evaluate(traj_gt, traj_pred)
		print('  RPE', rpe_tran, rpe_rot)
