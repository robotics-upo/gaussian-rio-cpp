#pragma once
#include "kalman_core.hpp"
#include "radar.hpp"
#include "imu.hpp"

#include <functional>

namespace upo_gaussians {

	namespace detail {

		struct RioBaseInitParams : StrapdownInitParams {
			Pose   radar_to_imu  = Pose::Identity();
			double match_pos_std = 1.0f;             ///< Default scan matching position uncertainty [m]
			double match_rot_std = 6.0*M_TAU/360.0;  ///< Default scan matching rotation uncertainty [rad]
		};

		struct RioInput {
			double          radar_time;
			RadarCloud      radar_scan;
			ImuData::Bundle imu_data;
		};

	}

	struct RioBase : public Strapdown {

		using InitParams = detail::RioBaseInitParams;
		using Input      = detail::RioInput;
		using Keyframer  = std::function<bool(RioBase const&)>;

		RioBase(
			Keyframer keyframer,
			InitParams const& p = InitParams{}
		);

		bool is_initial() const { return m_ref_time < 0.0; }
		double time() const { return m_imu_time - m_ref_time; }

		bool has_keyframe() const { return m_keyframe_time >= 0.0; }
		double kf_time() const { return time() - m_keyframe_time; }
		Pose kf_pose() const { return m_keyframe.inverse()*pose(); }

		void process(Input const& input);

	protected:
		Pose m_radar_to_imu;

		double m_ref_time = -1.0;
		double m_imu_time = -1.0;
		Vec<2> m_imu_rp = Vec<2>::Zero();
		Vec<3> m_angvel = Vec<3>::Zero();

		Pose m_keyframe = Pose::Identity();
		double m_keyframe_time = -1.0;
		Keyframer m_keyframer;

		double m_match_pos_cov;
		double m_match_rot_cov;

		void initialize(
			double time,
			Vec<3> const& vel,
			Mat<3> const& vel_cov
		) {
			m_ref_time = m_imu_time = time;
			init_vel(vel, vel_cov);
		}

		void update_scanmatch_3d(
			Pose const& match_pose,
			Vec<6> const& match_covdiag
		) {
			Strapdown::update_scanmatch_3d(m_keyframe, match_pose, match_covdiag);
		}

		void update_scanmatch_3d(Pose const& match_pose) {
			Vec<6> match_covdiag;
			match_covdiag.segment<3>(0).fill(m_match_pos_cov);
			match_covdiag.segment<3>(3).fill(m_match_rot_cov);
			update_scanmatch_3d(match_pose, match_covdiag);
		}

		virtual void scan_matching(RadarCloud const& cl) { }
		virtual bool process_keyframe(RadarCloud&&) { return false; }

	private:
		void process_imu(ImuData::Bundle const& bundle);
		RadarCloud process_egovel(RadarCloud const& cl, double time);
	};

}
