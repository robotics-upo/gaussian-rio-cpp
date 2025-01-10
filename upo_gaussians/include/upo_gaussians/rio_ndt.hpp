#pragma once
#include "rio_base.hpp"

namespace upo_gaussians {

	namespace detail {

		struct NdtParams {
			double xfrm_epsilon    = 0.01; ///< Minimum transformation difference for termination condition
			double step_size       = 0.1;  ///< Maximum step size for More-Thuente line search
			float  grid_resolution = 2.0f; ///< Resolution of NDT grid structure (VoxelGridCovariance)
			int    max_iters       = 35;   ///< Maximum number of registration iterations
		};

		struct RioNdtInitParams : RioBaseInitParams {
			NdtParams ndt_params{};
		};

	}

	struct RioNdt final : public RioBase {

		using InitParams = detail::RioNdtInitParams;
		using Input      = detail::RioInput;
		using Keyframer  = RioBase::Keyframer;
		using NdtParams  = detail::NdtParams;

		RioNdt(
			Keyframer keyframer,
			InitParams const& p = InitParams{}
		);

	protected:
		void scan_matching(RadarCloud const& cl);
		bool process_keyframe(RadarCloud&& cl);

	private:
		NdtParams m_ndt_params;
		RadarCloud::Ptr m_saved_cloud{};

	};

}
