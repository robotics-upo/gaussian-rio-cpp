#pragma once
#include "types.hpp"

namespace upo_gaussians {

	// Workaround for gcc/clang bug: https://stackoverflow.com/a/67637333
	namespace detail {

		struct StrapdownInitParams {
			double rp_att_std     = 15*M_TAU/360;
			double r2i_tran_std   = 0.05;
			double r2i_rot_std    = 1.0*M_TAU/360;
			double accel_bias_std = 0.02;
			double gyro_bias_std  = 0.00035*M_TAU/360;
		};

		struct StrapdownPropParams {
			double gravity          = 9.80511;
			double w_vel_std        = 0.1;
			double w_att_std        = 0.005;
			double w_accel_bias_std = 0.0;
			double w_gyro_bias_std  = 0.0;
		};

	};

	struct Strapdown {

		using InitParams = detail::StrapdownInitParams;
		using PropParams = detail::StrapdownPropParams;

		enum Fields {
			Pos        = 0,
			Vel        = Pos+3,
			R2ITran    = Vel+3,
			AccBias    = R2ITran+3,
			GyrBias    = AccBias+3,
			StateTotal = GyrBias+3,

			AttError   = StateTotal,
			R2IRotErr  = AttError+3,
			CovTotal   = R2IRotErr+3,
		};

		enum Noises {
			WVel     = 0,
			WAtt     = WVel+3,
			WAcc     = WAtt+3,
			WGyr     = WAcc+3,
			WAccBias = WGyr+3,
			WGyrBias = WAccBias+3,
			WTotal   = WGyrBias+3,
		};

		Vec<3> position()   const { return m_state.segment<3>(Pos);           }
		Quat   attitude()   const { return m_attitude;                        }
		Pose   pose()       const { return make_pose(attitude(), position()); }
		Vec<3> r2i_tran()   const { return m_state.segment<3>(R2ITran);       }
		Quat   r2i_rot()    const { return m_r2i_rot;                         }
		Pose   r2i_pose()   const { return make_pose(r2i_rot(), r2i_tran());  }
		Vec<3> velocity()   const { return m_state.segment<3>(Vel);           }
		Vec<3> accel_bias() const { return m_state.segment<3>(AccBias);       }
		Vec<3> gyro_bias()  const { return m_state.segment<3>(GyrBias);       }
		Mat<6> error_cov() const;

		Strapdown(
			InitParams const& p = InitParams{},
			Pose const& radar_to_imu = Pose::Identity(),
			bool want_yaw_gyro_bias = false
		);

		void init_vel(
			Vec<3> const& vel,
			Mat<3> const& vel_cov
		);

		void propagate_imu(
			double timediff,
			Vec<3> const& accel,
			Vec<3> const& accel_covdiag,
			Vec<3> const& gyro,
			Vec<3> const& gyro_covdiag,
			PropParams const& p = PropParams{}
		);

		Vec<3> calc_egovel(
			Vec<3> const& angvel = Vec<3>::Zero()
		) const;

		void update_egovel(
			Vec<3> const& egovel,
			Mat<3> const& egovel_cov,
			Vec<3> const& angvel = Vec<3>::Zero(),
			double outlier_percentile = 0.05
		);

		void update_scanmatch(
			Pose const& kf_pose,
			Mat<6> const& kf_cov,
			Pose const& match_pose,
			Vec<6> const& match_covdiag,
			bool full_6dof = false,
			double outlier_percentile = 0.1
		);

	private:
		using State = Vec<StateTotal>;
		using Cov = Mat<CovTotal>;

		State m_state    = State::Zero();
		Quat  m_attitude = Quat::Identity();
		Quat  m_r2i_rot  = Quat::Identity();
		Cov   m_cov      = Cov::Zero();

		auto padded_state() const {
			Vec<CovTotal> ret;
			ret.setZero();
			ret.segment<StateTotal>(0) = m_state;
			return ret;
		}

		template <int N, typename RType = Mat<N>>
		void update_common(
			const char* name,
			Vec<N> const& residual,
			RType const& R,
			Mat<N,CovTotal> const& H,
			double outlier_percentile
		);
	};
}
