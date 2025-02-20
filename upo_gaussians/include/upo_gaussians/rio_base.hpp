#pragma once
#include "kalman_core.hpp"
#include "radar.hpp"
#include "imu.hpp"

#include <functional>

namespace upo_gaussians {

	namespace detail {

		struct RioBaseInitParams : StrapdownInitParams, StrapdownPropParams {
			Pose     radar_to_imu  = Pose::Identity(); ///< Radar position/attitude with respect to body/IMU
			double   voxel_size    = 0.25;             ///< Voxel grid resolution used for downsampling (0=disable) [m]
			double   match_pos_std = 1.0f;             ///< Default scan matching position uncertainty [m]
			double   match_rot_std = 6.0*M_TAU/360.0;  ///< Default scan matching rotation uncertainty [rad]
			float    egovel_pct    = 0.05f;            ///< Egovelocity outlier percentile for EKF update rejection filter [0,1)
			float    scanmatch_pct = 0.1f;             ///< Scan matching outlier percentile for EKF update rejection filter [0,1)
			uint8_t  num_threads   = 4;                ///< Number of threads used for processing (hint)
			bool     match_6dof    = false;            ///< true for 6-DoF scan matching, false for 3-DoF (x/y/yaw)
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
		double match_time() const { return time() - m_match_time; }
		Pose kf_pose() const { return m_keyframe.inverse()*pose(); }

		Vec<3> egovel() const { return calc_egovel(m_angvel, m_radar_to_imu); }

		void process(Input const& input);

	protected:
		PropParams m_prop_params;
		Pose m_radar_to_imu;

		double m_ref_time = -1.0;
		double m_imu_time = -1.0;
		Vec<2> m_imu_rp = Vec<2>::Zero();
		Vec<3> m_angvel = Vec<3>::Zero();

		Pose m_keyframe = Pose::Identity();
		Mat<6> m_keyframe_cov{};
		double m_keyframe_time = -1.0;
		double m_match_time = -1.0;
		Keyframer m_keyframer;

		double m_match_pos_cov;
		double m_match_rot_cov;
		bool   m_match_6dof;

		unsigned m_num_threads;
		double   m_voxel_size;

		float m_egovel_pct;
		float m_scanmatch_pct;

		unsigned num_threads() const { return m_num_threads; }
		double   voxel_size()  const { return m_voxel_size;  }

		void initialize(
			double time,
			Vec<3> const& vel,
			Mat<3> const& vel_cov
		) {
			m_ref_time = m_imu_time = time;
			init_vel(vel, vel_cov);
		}

		void update_scanmatch(
			Pose const& match_pose,
			Vec<6> const& match_covdiag
		) {
			Strapdown::update_scanmatch(m_keyframe, m_keyframe_cov, match_pose, match_covdiag, m_match_6dof, m_scanmatch_pct);
		}

		void update_scanmatch(Pose const& match_pose) {
			Vec<6> match_covdiag;
			match_covdiag.segment<3>(0).fill(m_match_pos_cov);
			match_covdiag.segment<3>(3).fill(m_match_rot_cov);
			update_scanmatch(match_pose, match_covdiag);
		}

		virtual bool scan_matching(RadarCloud::Ptr) { return false; }
		virtual bool process_keyframe(RadarCloud::Ptr) { return false; }

	private:
		void process_imu(ImuData::Bundle const& bundle);
		RadarCloud process_egovel(RadarCloud const& cl, double time);
	};

}
