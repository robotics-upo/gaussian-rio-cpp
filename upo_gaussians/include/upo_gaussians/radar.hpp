#pragma once
#include "types.hpp"

namespace upo_gaussians {

	struct RadarPoint {
		PCL_ADD_POINT4D;
		float power;
		float doppler;
		float _pad[2];
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};

	using RadarCloud = pcl::PointCloud<RadarPoint>;

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

		unsigned calc_num_iters() const {
			if (num_iters) {
				return num_iters;
			} else {
				return 0.5f + logf(1.0f - p_success) / logf(1.0f - powf(1.0f-p_outlier, num_points));
			}
		}
	};

	struct EgoVelResult {
		Vec<3> egovel;
		Mat<3> egovel_cov;
		RadarCloud inliers;
	};

	bool calc_radar_egovel(
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
