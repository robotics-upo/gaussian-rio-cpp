#include <upo_gaussians/rio_base.hpp>
#include <pcl/common/transforms.h>
#include <iostream>

namespace upo_gaussians {

RioBase::RioBase(
	Keyframer keyframer,
	InitParams const& p
) : Strapdown{p, true},
	m_radar_to_imu{p.radar_to_imu},
	m_keyframer{std::move(keyframer)},
	m_match_pos_cov{p.match_pos_std*p.match_pos_std},
	m_match_rot_cov{p.match_rot_std*p.match_rot_std}
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

	if (has_keyframe()) {
		scan_matching(cl);
	}

	if (m_keyframer(*this) && process_keyframe(std::move(cl))) {
		m_keyframe      = pose();
		m_keyframe_time = time();
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

		propagate_imu(tdiff, imu.accel, imu.accel_covdiag, imu.gyro, imu.gyro_covdiag);

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
		update_egovel(r.egovel, r.egovel_cov, m_angvel, m_radar_to_imu);
	}

	return std::move(r.inliers);
}

}
