#pragma once
#include "types.hpp"

namespace upo_gaussians {

	struct EIGEN_ALIGN16 RadarPoint {
		PCL_ADD_POINT4D;
		float power;
		float doppler;
		float _pad[2];
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		constexpr RadarPoint(float x = 0.0f, float y = 0.0f, float z = 0.0f, float power = 0.0f, float doppler = 0.0f) :
			data{x,y,z,1.0f}, power{power}, doppler{doppler}, _pad{} { }
		constexpr RadarPoint(RadarPoint const&) = default;
		constexpr RadarPoint(RadarPoint&) = default;
	};

	struct EIGEN_ALIGN16 PosedLocalPoint {
		PCL_ADD_POINT4D;
		pcl::PointXYZ local_pos;
		Quatf local_rot;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		PosedLocalPoint(float x = 0.0f, float y = 0.0f, float z = 0.0f) :
			data{x,y,z,1.0f}, local_pos{}, local_rot{Quatf::Identity()} { }
		PosedLocalPoint(PosedLocalPoint const&) = default;
		PosedLocalPoint(PosedLocalPoint&) = default;

		Eigen::Isometry3f getLocalPose() const {
			Eigen::Isometry3f ret = Eigen::Isometry3f::Identity();
			ret.matrix().block(0,0,3,3) = local_rot.toRotationMatrix();
			ret.matrix().block(0,3,3,1) = local_pos.getArray3fMap();
			return ret;
		}

	};

	using RadarCloud = pcl::PointCloud<RadarPoint>;
	using IncidenceCloud = pcl::PointCloud<PosedLocalPoint>;

	struct RadarFilterParams {
		float min_power     = 10.0f;
		float max_azimuth   = 56.5f*M_TAU/360;
		float max_elevation = 22.5f*M_TAU/360;
	};

	RadarCloud filter_radar_cloud(
		RadarCloud const& in_cloud,
		RadarFilterParams const& p = RadarFilterParams{}
	);

	struct EgoVelParams {
		uint16_t num_points    = 5;
		uint16_t num_iters     = 0;
		float    still_std     = 0.02f;
		float    still_thresh  = 0.05f;
		float    inlier_thresh = 0.3f;
		float    min_inlier_p  = 0.5f;
		float    p_outlier     = 0.05f;
		float    p_success     = 0.995f;
		double   moving_std    = 0.1;

		unsigned calc_num_iters() const {
			if (num_iters) {
				return num_iters;
			} else {
				return 0.5f + logf(1.0f - p_success) / logf(1.0f - powf(1.0f-p_outlier, num_points));
			}
		}
	};

	enum class EgoVelState {
		Fail  = 0,
		Still = 1,
		Ok    = 2,
	};

	struct EgoVelResult {
		Vec<3> egovel;
		Mat<3> egovel_cov;
		RadarCloud inliers;
	};

	EgoVelState calc_radar_egovel(
		EgoVelResult& ret,
		RadarCloud const& in_cloud,
		EgoVelParams const& p = EgoVelParams{}
	);

}

#define UPO_GAUSSIANS_REGISTER_RADAR_POINT_STRUCT(_power, _doppler) \
	POINT_CLOUD_REGISTER_POINT_STRUCT(upo_gaussians::RadarPoint, \
		(float, x,       x) \
		(float, y,       y) \
		(float, z,       z) \
		(float, power,   _power) \
		(float, doppler, _doppler) \
	)
