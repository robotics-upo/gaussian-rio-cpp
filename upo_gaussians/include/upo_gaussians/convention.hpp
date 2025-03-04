#pragma once
#include "types.hpp"

namespace upo_gaussians {

	enum class Convention {
		NWU, // ROS/Default convention
		NED, // Aerospace convention
		NDW,
		NUE,
		ENU, // Geography convention
		ESD,
		EUS, // Geometry convention
		EDN, // Camera convention
	};

	template <typename T>
	constexpr Vec<3,T> decode_coords(T const& x, T const& y, T const& z, Convention c = Convention::NWU)
	{
		Vec<3,T> ret;
		switch (c) {
			default:
			case Convention::NWU: ret << +x,+y,+z; break;
			case Convention::NED: ret << +x,-y,-z; break;
			case Convention::NDW: ret << +x,+z,-y; break;
			case Convention::NUE: ret << +x,-z,+y; break;
			case Convention::ENU: ret << +y,-x,+z; break;
			case Convention::ESD: ret << -y,-x,-z; break;
			case Convention::EUS: ret << -z,-x,+y; break;
			case Convention::EDN: ret << +z,-x,-y; break;
		}
		return ret;
	}

	template <typename PointType>
	void apply_convention_in_place(pcl::PointCloud<PointType>& cl, Convention c)
	{
		if (c == Convention::NWU) {
			// No need to do anything
			return;
		}

		auto xyz = cl.getMatrixXfMap();
		for (Eigen::Index i = 0; i < xyz.cols(); i ++) {
			xyz.col(i).template segment<3>(0) = decode_coords(xyz(0,i), xyz(1,i), xyz(2,i), c);
		}
	}

}
