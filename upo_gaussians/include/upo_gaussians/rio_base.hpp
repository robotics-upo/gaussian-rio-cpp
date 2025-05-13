#pragma once
#include "kalman_core.hpp"
#include "radar.hpp"
#include "imu.hpp"

#include <functional>

namespace upo_gaussians {

	namespace detail {

		struct RioBaseInitParams : StrapdownPropParams {
			Pose     radar_to_imu  = Pose::Identity(); ///< Radar position/attitude with respect to body/IMU
			double   max_init_time = 3.0;              ///< Maximum initialization time [s]
			double   r2i_tran_std  = 0.0;              ///< Radar-to-IMU position uncertainty [m] (was: 0.05)
			double   r2i_rot_std   = 0.5*M_TAU/360;    ///< Radar-to-IMU rotation uncertainty [m] (was: 1.0)
			double   voxel_size    = 0.25;             ///< Voxel grid resolution used for downsampling (0=disable) [m]
			double   match_pos_std = 0.5f;             ///< Default scan matching position uncertainty [m]
			double   match_rot_std = 2.0f*M_TAU/360.0; ///< Default scan matching rotation uncertainty [rad]
			float    egovel_pct    = 0.05f;            ///< Egovelocity outlier percentile for EKF update rejection filter [0,1)
			float    scanmatch_pct = 0.1f;             ///< Scan matching outlier percentile for EKF update rejection filter [0,1)
			uint8_t  num_threads   = 4;                ///< Number of threads used for processing (hint)
			bool     deterministic = true;             ///< Attempts to use deterministic algorithms, may be slower (hint)
			bool     match_6dof    = false;            ///< true for 6-DoF scan matching, false for 3-DoF (x/y/yaw)
			bool     filter_cloud  = false;            ///< Whether to filter the radar cloud before processing
			StrapdownInitImuParams init_imu_params;    ///< IMU-based initialization parameters
		};

		struct RioInput {
			double          radar_time;
			RadarCloud      radar_scan;
			ImuData::Bundle imu_data;
		};

	}

	struct RioBase : public Strapdown {

		using InitImuParams = detail::StrapdownInitImuParams;
		using InitParams = detail::RioBaseInitParams;
		using Input      = detail::RioInput;
		using Keyframer  = std::function<bool(RioBase const&)>;

		RioBase(
			Keyframer keyframer,
			InitParams const& p = InitParams{}
		);

		bool is_initial() const { return m_imu_time < 0.0; }
		double time() const { return m_imu_time - m_ref_time; }

		bool has_keyframe() const { return m_keyframe_time >= 0.0; }
		double kf_time() const { return time() - m_keyframe_time; }
		double match_time() const { return time() - m_match_time; }
		Pose kf_pose() const { return m_keyframe.inverse()*pose(); }

		Vec<3> egovel() const { return calc_egovel(m_angvel); }

		void process(Input const& input);

	protected:
		PropParams m_prop_params;
		InitImuParams m_init_imu_params;

		double m_max_init_time;
		double m_ref_time = -1.0;
		double m_imu_time = -1.0;
		double m_init_time = -1.0;
		Vec<3> m_imu_init_accel = Vec<3>::Zero();
		Vec<3> m_imu_init_gyro  = Vec<3>::Zero();
		size_t m_imu_init_num   = 0;
		Vec<3> m_angvel = Vec<3>::Zero();

		Pose m_keyframe = Pose::Identity();
		Mat<6> m_keyframe_cov{};
		double m_initial_kf_time = -1.0;
		double m_keyframe_time = -1.0;
		double m_match_time = -1.0;
		Keyframer m_keyframer;

		double m_match_pos_cov;
		double m_match_rot_cov;
		bool   m_deterministic;
		bool   m_match_6dof;
		bool   m_filter_cloud;

		unsigned m_num_threads;
		double   m_voxel_size;

		float m_egovel_pct;
		float m_scanmatch_pct;

		unsigned num_threads() const { return m_num_threads; }
		bool   deterministic() const { return m_deterministic; }
		double   voxel_size()  const { return m_voxel_size;  }

		void init_imu() {
			m_imu_time = m_init_time;

			if (m_imu_init_num) {
				Strapdown::init_imu(
					m_imu_init_accel/m_imu_init_num,
					m_imu_init_gyro/m_imu_init_num,
					m_prop_params.gravity,
					m_init_imu_params
				);
			} else {
				// TODO: figure out what to do with no still IMU samples!
			}
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

		void commit_keyframe(double kf_time) {
			m_keyframe      = pose();
			m_keyframe_cov  = error_cov();
			m_keyframe_time = m_match_time = kf_time;
		}

	private:
		void process_imu_initial(ImuData::Bundle const& bundle);
		void process_imu(ImuData::Bundle const& bundle);
	};

}
