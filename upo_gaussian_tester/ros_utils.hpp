#pragma once
#include "config.hpp"

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

UPO_GAUSSIANS_REGISTER_RADAR_POINT_STRUCT(Power, Doppler)

namespace upo_gaussians {

	static inline void fix_radar_cloud_fields(sensor_msgs::PointCloud2& cl, Dataset::RadarConfig const& radar)
	{
		decltype(cl.fields) newfields;

		for (auto& f : cl.fields) {
			bool wantit = false;
			if (f.name == "x" || f.name == "y" || f.name == "z") {
				wantit = true;
			} else if (f.name == radar.field_power) {
				f.name = "Power";
				wantit = true;
			} else if (f.name == radar.field_doppler) {
				f.name = "Doppler";
				wantit = true;
			}
			if (wantit) {
				newfields.push_back(f);
			}
		}

		cl.fields = std::move(newfields);
	}

	static inline sensor_msgs::PointCloud2::Ptr instantiate_bag_message_as_cloud(rosbag::MessageInstance const& msg)
	{
		auto cloud2 = msg.instantiate<sensor_msgs::PointCloud2>();
		if (!cloud2) {
			auto cloud1 = msg.instantiate<sensor_msgs::PointCloud>();
			if (cloud1) {
				cloud2 = boost::make_shared<sensor_msgs::PointCloud2>();
				sensor_msgs::convertPointCloudToPointCloud2(*cloud1, *cloud2);
			}
		}

		return cloud2;
	}

	static inline bool rosbag_msg_to_radar_cloud(RadarCloud& out_cl, double& out_time, rosbag::MessageInstance const& msg, Dataset::RadarConfig const& radar)
	{
		auto roscl = instantiate_bag_message_as_cloud(msg);
		if (!roscl) {
			return false;
		}

		out_time = roscl->header.stamp.toSec();
		fix_radar_cloud_fields(*roscl, radar);
		pcl::moveFromROSMsg(*roscl, out_cl);
		apply_convention_in_place(out_cl, radar.convention);
		return true;
	}

}
