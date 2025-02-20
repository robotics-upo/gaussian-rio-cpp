#include <upo_gaussians/rio_base.hpp>
#include <pcl/common/transforms.h>
#include <iostream>

#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/util/downsampling_omp.hpp>

namespace upo_gaussians {

RioBase::RioBase(
	Keyframer keyframer,
	InitParams const& p
) : Strapdown{p, true},
	m_prop_params{p},
	m_radar_to_imu{p.radar_to_imu},
	m_keyframer{std::move(keyframer)},
	m_match_pos_cov{p.match_pos_std*p.match_pos_std},
	m_match_rot_cov{p.match_rot_std*p.match_rot_std},
	m_match_6dof{p.match_6dof},
	m_num_threads{p.num_threads},
	m_voxel_size{p.voxel_size},
	m_egovel_pct{p.egovel_pct},
	m_scanmatch_pct{p.scanmatch_pct}
{
}

void RioBase::process(Input const& input)
{
	if (!is_initial() && input.imu_data.size() != 0) {
		process_imu(input.imu_data);
	}

	auto cl = filter_radar_cloud(input.radar_scan);
	cl = process_egovel(cl, input.radar_time);

	pcl::transformPointCloud(cl, cl, m_radar_to_imu.matrix().cast<float>());

	if (m_voxel_size > 0.0) {
		cl = std::move(*small_gicp::voxelgrid_sampling_omp(cl, m_voxel_size, m_num_threads));
	}

	auto cl_ptr = std::make_shared<RadarCloud>(std::move(cl));

	if (has_keyframe() && scan_matching(cl_ptr)) {
		m_match_time = time();
	}

	if (m_keyframer(*this) && process_keyframe(cl_ptr)) {
		m_keyframe      = pose();
		m_keyframe_cov  = error_cov();
		m_keyframe_time = m_match_time = time();
	}
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

		propagate_imu(tdiff, imu.accel, imu.accel_covdiag, imu.gyro, imu.gyro_covdiag, m_prop_params);

		++num_ok;
		m_imu_time = imu.timestamp;
		mean_gyro += imu.gyro;
	}

	m_angvel = mean_gyro/num_ok - gyro_bias();
}

inline RadarCloud RioBase::process_egovel(RadarCloud const& cl, double time)
{
	EgoVelResult r;
	if (!calc_radar_egovel(r, cl)) {
		return cl;
	}

	if (is_initial()) {
		initialize(time, r.egovel, r.egovel_cov);
	} else {
		update_egovel(r.egovel, r.egovel_cov, m_angvel, m_radar_to_imu, m_egovel_pct);
	}

	return std::move(r.inliers);
}

}
