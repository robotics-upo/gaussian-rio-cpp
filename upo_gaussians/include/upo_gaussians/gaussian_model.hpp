#pragma once
#include "types.hpp"
#include "radar.hpp"

namespace upo_gaussians {

	namespace detail {

		struct GaussianFitParams {
			size_t   num_gaussians  = 150;
			double   initial_scale  = 1.0;
			double   min_scale      = 0.05;
			double   disc_thickness = 0.15;
			uint16_t num_threads    = 4;
			bool     verbose        = false;
		};

	}

	struct GaussianModel {
		using QuatArray = Eigen::Vector<Eigen::Quaterniond, Eigen::Dynamic>;
		using FitParams = detail::GaussianFitParams;

		VecArray<3> centers;
		VecArray<3> log_scales;
		QuatArray   quats;

		auto quats_array() {
			return Eigen::Map<VecArray<4>>{ quats.data()->coeffs().data(), Eigen::NoChange, quats.rows() };
		}

		auto quats_array() const {
			return Eigen::Map<VecArray<4> const>{ quats.data()->coeffs().data(), Eigen::NoChange, quats.rows() };
		}

		void fit(AnyCloudIn cl, FitParams const& p);

		template <typename PointType>
		void fit(pcl::PointCloud<PointType> const& cl, FitParams const& p = FitParams{}) {
			fit(cl.getMatrixXfMap(), p);
		}
	};

}
