#pragma once
#include "rio_base.hpp"
#include "gaussian_model.hpp"

namespace upo_gaussians {

	namespace detail {

		struct RioGaussianInitParams : RioBaseInitParams {
			size_t gaussian_size = 20;
			size_t num_particles = 4;
			double match_thresh = 50.0;
			double particle_std_xyz = 1.0;
			double particle_std_rot = 6.0*M_TAU/360.0;
		};

	}

	struct RioGaussian final : public RioBase {
		using InitParams = detail::RioGaussianInitParams;
		using Input      = detail::RioInput;
		using Keyframer  = RioBase::Keyframer;

		RioGaussian(
			Keyframer keyframer,
			InitParams const& p = InitParams{}
		);

		~RioGaussian();

		/* inline */ PoseArray particle_swarm();

	protected:
		bool scan_matching(RadarCloud::Ptr cl);
		bool process_keyframe(RadarCloud::Ptr cl);

	private:
		size_t m_gaussian_sz;
		size_t m_num_particles;
		double m_match_thresh;
		double m_particle_std_xyz;
		double m_particle_std_rot;

		GaussianModel m_model;
		double m_basemahal;
	};

}
