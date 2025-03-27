#include <upo_gaussians/rio_base.hpp>
#include <pcl/common/transforms.h>
#include <iostream>

#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>

namespace upo_gaussians {

RioBase::RioBase(
	Keyframer keyframer,
	InitParams const& p
) :
	m_prop_params{p},
	m_init_imu_params{p.init_imu_params},
	m_max_init_time{p.max_init_time},
	m_keyframer{std::move(keyframer)},
	m_match_pos_cov{p.match_pos_std*p.match_pos_std},
	m_match_rot_cov{p.match_rot_std*p.match_rot_std},
	m_deterministic{p.deterministic},
	m_match_6dof{p.match_6dof},
	m_filter_cloud{p.filter_cloud},
	m_num_threads{p.num_threads},
	m_voxel_size{p.voxel_size},
	m_egovel_pct{p.egovel_pct},
	m_scanmatch_pct{p.scanmatch_pct}
{
	init_r2i(p.radar_to_imu, p.r2i_tran_std, p.r2i_rot_std);
}

void RioBase::process(Input const& input)
{
	RadarCloud cl;
	auto const* cl_input = &input.radar_scan;
	if (m_filter_cloud) {
		cl = filter_radar_cloud(*cl_input);
		cl_input = &cl;
	}

	EgoVelResult r;
	auto st = calc_radar_egovel(r, *cl_input);
	if (st != EgoVelState::Fail) {
		cl_input = &r.inliers;
	}

	pcl::transformPointCloud(*cl_input, cl, r2i_pose().matrix().cast<float>());

	if (m_voxel_size > 0.0) {
		if (m_deterministic) {
			cl = std::move(*small_gicp::voxelgrid_sampling(cl, m_voxel_size));
		} else {
			cl = std::move(*small_gicp::voxelgrid_sampling_omp(cl, m_voxel_size, m_num_threads));
		}
	}

	auto cl_ptr = std::make_shared<RadarCloud>(std::move(cl));

	bool has_imu = input.imu_data.size() != 0;
	if (is_initial()) {
		if (st == EgoVelState::Still) do {
			if (m_ref_time < 0.0) {
				m_ref_time = input.radar_time;
			}

			if (has_imu) {
				// Promote Still to Ok if the time threshold is crossed
				if (m_imu_init_num && (input.radar_time - m_ref_time) >= m_max_init_time) {
					st = EgoVelState::Ok;
					break;
				}

				// Process IMU messages
				process_imu_initial(input.imu_data);
				m_init_time = input.radar_time;
			}

			// Attempt to capture a keyframe while the robot is still
			if (m_initial_kf_time < 0.0 && process_keyframe(cl_ptr)) {
				m_initial_kf_time = input.radar_time;
			}
		} while (0);

		// Exit early if we haven't detected movement
		if (st != EgoVelState::Ok || !has_imu) {
			return;
		}

		// Initialize EKF state using accumulated IMU data
		init_imu();
		//init_vel();

		// Commit the initial keyframe
		if (m_initial_kf_time >= 0.0) {
			commit_keyframe(m_initial_kf_time);
		}

		// Initialize egovelocity
		//init_egovel(r.egovel, r.egovel_cov);
	} else if (!has_imu) {
		// Ignore this scan if no IMU messages were received
		return;
	}

	process_imu(input.imu_data);

	if (st != EgoVelState::Fail) {
		update_egovel(r.egovel, r.egovel_cov, m_angvel, m_egovel_pct);
	}

	if (has_keyframe() && scan_matching(cl_ptr)) {
		m_match_time = time();
	}

	if (m_keyframer(*this) && process_keyframe(cl_ptr)) {
		commit_keyframe(time());
	}
}

inline void RioBase::process_imu_initial(ImuData::Bundle const& bundle)
{
	for (auto const& imu : bundle) {
		m_imu_init_accel += imu.accel;
		m_imu_init_gyro  += imu.gyro;
	}

	m_imu_init_num += bundle.size();
}

inline void RioBase::process_imu(ImuData::Bundle const& bundle)
{
	Vec<3> mean_gyro = Vec<3>::Zero();
	size_t num_ok = 0;

	for (auto const& imu : bundle) {
		double tdiff = imu.timestamp - m_imu_time;
		if (tdiff < 0) {
			std::cerr << "  [RIO] IMU messages are out of order!" << std::endl;
			continue;
		}

		if (is_initial()) {
			m_imu_init_accel += imu.accel;
			m_imu_init_gyro  += imu.gyro;
			m_imu_init_num++;
		} else {
			propagate_imu(tdiff, imu.accel, imu.accel_covdiag, imu.gyro, imu.gyro_covdiag, m_prop_params);
		}

		++num_ok;
		m_imu_time = imu.timestamp;
		mean_gyro += imu.gyro;
	}

	m_angvel = mean_gyro/num_ok - gyro_bias();
}

}
