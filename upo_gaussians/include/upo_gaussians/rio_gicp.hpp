#pragma once
#include "rio_base.hpp"

namespace upo_gaussians {

	namespace detail {

		struct GicpParams {
			unsigned num_neighbors = 10;
			unsigned max_iters     = 20;
			double   max_corr_dist = 1.0;
			double   tran_eps      = 1.0e-3;
			double   rot_eps       = 0.1*M_TAU/360.0;
		};

		struct RioGicpInitParams : RioBaseInitParams {
			GicpParams gicp_params{};
		};

		struct RioGicpModelTraits;

	}

	struct RioGicp final : public RioBase {

		using InitParams = detail::RioGicpInitParams;
		using Input      = detail::RioInput;
		using Keyframer  = RioBase::Keyframer;
		using GicpParams = detail::GicpParams;

		RioGicp(
			Keyframer keyframer,
			InitParams const& p = InitParams{}
		);

		~RioGicp();

	protected:
		bool scan_matching(RadarCloud::Ptr cl);
		bool process_keyframe(RadarCloud::Ptr cl);

	private:
		GicpParams m_gicp_params;

		class Model;
		friend struct detail::RioGicpModelTraits;

		std::unique_ptr<Model> m_model{};
		std::unique_ptr<Model> m_cached_model{};
	};

}
