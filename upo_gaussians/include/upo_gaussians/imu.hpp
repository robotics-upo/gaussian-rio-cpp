#pragma once
#include "types.hpp"
#include "convention.hpp"

#if __has_include(<sensor_msgs/Imu.h>)
#include <sensor_msgs/Imu.h>
#define _UPO_GAUSSIANS_ROS_TYPE ::sensor_msgs::Imu
#endif

namespace upo_gaussians {

	namespace detail {

		struct ImuDecodeParams {
			double     accel_std  = 0.01;
			double     gyro_std   = 0.01;
			Convention convention = Convention::NWU;
		};

	}

	struct ImuData {
		using Bundle = std::vector<ImuData>;
		using DecodeParams = detail::ImuDecodeParams;

		double timestamp;
		Vec<3> accel;
		Vec<3> gyro;
		Vec<3> accel_covdiag;
		Vec<3> gyro_covdiag;

#ifdef _UPO_GAUSSIANS_ROS_TYPE
		static ImuData fromROS(
			_UPO_GAUSSIANS_ROS_TYPE const& ros,
			DecodeParams const& p = DecodeParams{}
		) {
			ImuData ret;
			ret.timestamp = ros.header.stamp.toSec();
			ret.accel = decode_coords(ros.linear_acceleration.x, ros.linear_acceleration.y, ros.linear_acceleration.z, p.convention);
			ret.gyro  = decode_coords(ros.angular_velocity.x,    ros.angular_velocity.y,    ros.angular_velocity.z,    p.convention);
			ret.accel_covdiag.fill(p.accel_std*p.accel_std);
			ret.gyro_covdiag.fill(p.gyro_std*p.gyro_std);
			return ret;
		}
#endif
	};

}
