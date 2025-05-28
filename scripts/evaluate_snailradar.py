import numpy as np

from evo.core import sync
from evo.core.units import Unit
from evo.core.trajectory import PoseTrajectory3D
from evo.core.metrics import RPE, PoseRelation, StatisticsType

from typing import Tuple, List

TAU = float(2*np.pi)

DATASET_DIR = '/mnt/data/work/datasets/snailradar'
TRAJDIR = './output/snail-radar'

DATASET_SEQS = ['st_20231213_1', 'iaf_20231213_2', 'iaf_20231213_3', 'if_20231213_4', 'if_20231213_5']
METHOD_NAMES = ['ndt', 'gicp', 'vgicp', 'gaussian_p1', 'gaussian_p8']
RADAR_NAMES  = ['ars', 'eagle']

all_results = np.zeros((len(METHOD_NAMES), len(DATASET_SEQS), len(RADAR_NAMES), 2), dtype=np.float64)

def quat_conj(q:np.ndarray):
	assert q.shape[-1] == 4
	ret = -q
	ret[...,0] = -ret[...,0]
	return ret

def quat_vec_mult(q:np.ndarray, v:np.ndarray):
	qv = q[...,1:4]
	qw = q[...,0,None]
	uv = np.cross(qv, v)
	uuv = np.cross(qv, uv)
	return v + 2*(qw*uv + uuv)

def quat_mult(q0:np.ndarray, q1:np.ndarray) -> np.ndarray:
	w0,x0,y0,z0 = q0[...,0], q0[...,1], q0[...,2], q0[...,3]
	w1,x1,y1,z1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
	return np.stack([
		w0*w1 - x0*x1 - y0*y1 - z0*z1,
		w0*x1 + x0*w1 + y0*z1 - z0*y1,
		w0*y1 + y0*w1 + z0*x1 - x0*z1,
		w0*z1 + z0*w1 + x0*y1 - y0*x1,
	], axis=-1)

def quat_to_rpy(quat: np.ndarray):
	qw,qx,qy,qz = quat[...,0], quat[...,1], quat[...,2], quat[...,3]

	# roll (x-axis rotation)
	sinr_cosp = 2*(qw*qx + qy*qz)
	cosr_cosp = 1 - 2*(qx*qx + qy*qy)
	roll = np.arctan2(sinr_cosp, cosr_cosp)

	# pitch (y-axis rotation)
	sinp = np.sqrt(1 + 2*(qw*qy - qx*qz))
	cosp = np.sqrt(1 - 2*(qw*qy - qx*qz))
	pitch = 2*np.arctan2(sinp, cosp) - TAU/4

	# yaw (z-axis rotation)
	siny_cosp = 2*(qw*qz + qx*qy)
	cosy_cosp = 1 - 2*(qy*qy + qz*qz)
	yaw = np.arctan2(siny_cosp, cosy_cosp)

	return np.stack((roll,pitch,yaw), axis=-1)

def rpy_to_quat(rpy: np.ndarray):
	r,p,y = rpy[...,0]/2, rpy[...,1]/2, rpy[...,2]/2

	cosr,sinr = np.cos(r),np.sin(r)
	cosp,sinp = np.cos(p),np.sin(p)
	cosy,siny = np.cos(y),np.sin(y)

	return np.stack([
		cosr*cosp*cosy + sinr*sinp*siny,
		sinr*cosp*cosy - cosr*sinp*siny,
		cosr*sinp*cosy + sinr*cosp*siny,
		cosr*cosp*siny - sinr*sinp*cosy,
	], axis=-1)

def mat_to_quat(mat: np.ndarray) -> np.ndarray:
	tx,ty,tz = mat[0,0],mat[1,1],mat[2,2]
	if tz <= 0:
		if tx >= ty:
			x = 1 + tx - ty - tz
			y = mat[1,0] + mat[0,1]
			z = mat[2,0] + mat[0,2]
			w = mat[2,1] - mat[1,2]
		else:
			x = mat[1,0] + mat[0,1]
			y = 1 - tx + ty - tz
			z = mat[2,1] + mat[1,2]
			w = mat[0,2] - mat[2,0]
	else:
		if -tx >= ty:
			x = mat[2,0] + mat[0,2]
			y = mat[2,1] + mat[1,2]
			z = 1 - tx - ty + tz
			w = mat[1,0] - mat[0,1]
		else:
			x = mat[2,1] - mat[1,2]
			y = mat[0,2] - mat[2,0]
			z = mat[1,0] - mat[0,1]
			w = 1 + tx + ty + tz
	q = np.stack([w,x,y,z], axis=-1)
	return q / np.linalg.norm(q)

R_xt32_to_body = mat_to_quat(np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=np.float64))
R_body_to_xt32 = quat_conj(R_xt32_to_body)
R_utm50r_to_world = mat_to_quat(np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=np.float64))

def load_tum_traj(fname):
	mat = np.loadtxt(fname, dtype=np.float64, delimiter=' ', comments='#')
	ts   = mat[:,0]
	pos  = mat[:,1:4]
	quat = np.roll(mat[:,4:8], 1, axis=1)
	return ts, pos, quat

def load_gt(seq):
	seq = seq.split('_')
	ts, pos, quat = load_tum_traj(f'{DATASET_DIR}/full_trajs/{seq[1]}/data{seq[2]}/utm50r_T_xt32.txt')
	# At this point, (pos,quat) = T_xt32_to_utm50r

	# T_body_to_utm50r = T_xt32_to_utm50r * T_body_to_xt32
	quat = quat_mult(quat, R_body_to_xt32[None])

	# T_body_to_world = R_utm50r_to_world * T_body_to_utm50r
	quat = quat_mult(R_utm50r_to_world, quat)
	pos = quat_vec_mult(R_utm50r_to_world, pos)

	return ts, pos, quat

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
			m_rot  = rpe_metric(traj_gt, traj_pred, PoseRelation.rotation_angle_rad, delta)*1000
		except:
			continue

		accum_tran.append(m_tran)
		accum_rot.append(m_rot)

	accum_tran = np.mean(accum_tran)
	accum_rot  = np.mean(accum_rot)

	return float(accum_tran), float(accum_rot)

for idx_seq, seqname in enumerate(DATASET_SEQS):
	print('Processing', seqname)
	gt_ts, gt_pos, gt_quat = load_gt(seqname)

	for idx_radar, radar in enumerate(RADAR_NAMES):
		for idx_method, method in enumerate(METHOD_NAMES):
			print(f'  ({method}, {radar})')
			try:
				pred_ts, pred_pos, pred_quat = load_tum_traj(f'{TRAJDIR}/{radar}_{method}_traj_{seqname}.txt')
			except IOError:
				continue

			# Align sequences
			if gt_ts[0] <= pred_ts[0]:
				ref_time = pred_ts[0]
				pred_id = 0
				gt_id = np.argmin(np.abs(gt_ts - ref_time))
			else:
				ref_time = gt_ts[0]
				pred_id = np.argmin(np.abs(pred_ts - ref_time))
				gt_id = 0

			gt_align_pos = gt_pos + (pred_pos[pred_id] - gt_pos[gt_id])
			gt_deyaw = rpy_to_quat(np.array([0.0, 0.0, -quat_to_rpy(gt_quat[gt_id])[2]], dtype=np.float64))
			gt_align_quat = quat_mult(gt_deyaw, gt_quat)
			gt_align_pos = quat_vec_mult(gt_deyaw, gt_align_pos)

			gt_traj   = PoseTrajectory3D(gt_align_pos, gt_align_quat, gt_ts)
			pred_traj = PoseTrajectory3D(pred_pos,     pred_quat,     pred_ts)

			rpe_tran, rpe_rot = evaluate(gt_traj, pred_traj)
			print('    RPE', rpe_tran, rpe_rot)

			all_results[idx_method, idx_seq, idx_radar, 0] = rpe_tran
			all_results[idx_method, idx_seq, idx_radar, 1] = rpe_rot

all_results = np.reshape(all_results, (len(METHOD_NAMES), -1))
#np.savetxt(f'{TRAJDIR}/saved_metrics.txt', all_results, delimiter='\t')
