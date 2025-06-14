#include <upo_gaussians/rio_gaussian.hpp>
#include <iostream>

namespace upo_gaussians {

RioGaussian::RioGaussian(
	Keyframer keyframer,
	InitParams const& p
) : RioBase{keyframer,p},
	m_gaussian_sz{p.gaussian_size},
	m_num_particles{p.num_particles},
	m_match_thresh{p.match_thresh},
	m_particle_std_xyz{p.particle_std_xyz},
	m_particle_std_rot{p.particle_std_rot}
{
	const char* fnp = getenv("FORCE_NUM_PARTICLES");
	if (fnp) {
		m_num_particles = atoi(fnp);
	}
}

RioGaussian::~RioGaussian() = default;

inline PoseArray RioGaussian::particle_swarm()
{
	PoseArray pa;
	pa.resize(m_num_particles);
	pa(0) = Pose::Identity();

	std::mt19937 m_rng{3135134162};
	std::normal_distribution rng;
	for (size_t i = 1; i < m_num_particles; i ++) {
		Vec<3> tran { rng(m_rng), rng(m_rng), rng(m_rng) };
		Vec<3> rot  { rng(m_rng), rng(m_rng), rng(m_rng) };
		pa(i) = make_pose(so3_exp(rot*m_particle_std_rot), tran*m_particle_std_xyz);
	}

	return pa;
}

bool RioGaussian::scan_matching(RadarCloud::Ptr cl)
{
	detail::GaussianMatchResults ret;
	bool ok = m_model.match(ret, *cl, kf_pose(), particle_swarm());
	if (!ok) {
		std::cerr << "  [RioGaussian] Warning: Not converged" << std::endl;
		return false;
	}

	auto L = ret.score - m_basemahal;
	if (L >= m_match_thresh) {
		std::cerr << "  [RioGaussian] Warning: Loss above threshold: " << L << std::endl;
		//return false;
	}

	update_scanmatch(ret.pose);
	return true;
}

bool RioGaussian::process_keyframe(RadarCloud::Ptr cl)
{
	detail::GaussianFitParams p;

	p.num_gaussians = (2*cl->size() + 1) / (2*m_gaussian_sz);
	p.num_threads = num_threads();
	//p.verbose = true;

	const char* fng = getenv("FORCE_NUM_GAUSSIANS");
	if (fng) {
		p.num_gaussians = atoi(fng);
	}

	std::cout << "RioGaussian: modeling " << p.num_gaussians << " gaussians with " << p.num_threads << " threads" << std::endl;

	m_model.fit_server(*cl, p);

	detail::GaussianMatchResults ret;
	if (!m_model.match(ret, *cl)) {
		std::cerr << "  [RioGaussian] Warning: Fitting returned bad model" << std::endl;
		return false;
	}

	m_basemahal = ret.score;

	return true;
}

}
