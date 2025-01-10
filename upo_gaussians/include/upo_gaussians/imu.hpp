#pragma once
#include "types.hpp"

#if __has_include(<sensor_msgs/Imu.h>)
#include <sensor_msgs/Imu.h>
#define _UPO_GAUSSIANS_ROS_TYPE ::sensor_msgs::Imu
#endif

namespace upo_gaussians {

	struct ImuData {
		using Bundle = std::vector<ImuData>;

		double timestamp;
		Vec<3> accel;
		Vec<3> gyro;
		Vec<3> accel_covdiag;
		Vec<3> gyro_covdiag;

#ifdef _UPO_GAUSSIANS_ROS_TYPE
		static ImuData fromROS(_UPO_GAUSSIANS_ROS_TYPE const& ros) {
			ImuData ret;
			ret.timestamp = ros.header.stamp.toSec();
			ret.accel << ros.linear_acceleration.x, -ros.linear_acceleration.y, -ros.linear_acceleration.z;
			ret.gyro  << ros.angular_velocity.x,    -ros.angular_velocity.y,    -ros.angular_velocity.z;
			ret.accel_covdiag.fill(0.0022281160035059417);
			ret.gyro_covdiag.fill(0.00011667951042710442);
			return ret;
		}
#endif
	};

}
