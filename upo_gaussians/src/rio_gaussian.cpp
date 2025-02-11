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
}

RioGaussian::~RioGaussian() = default;

inline PoseArray RioGaussian::particle_swarm()
{
	PoseArray pa;
	pa.resize(m_num_particles);
	pa(0) = Pose::Identity();

	std::normal_distribution rng;
	for (size_t i = 1; i < m_num_particles; i ++) {
		Vec<3> tran { rng(m_rng), rng(m_rng), rng(m_rng) };
		Vec<3> rot  { rng(m_rng), rng(m_rng), rng(m_rng) };
		pa(i) = make_pose(pure_quat_exp(0.5*rot*m_particle_std_rot), tran*m_particle_std_xyz);
	}

	return pa;
}

bool RioGaussian::scan_matching(RadarCloud::Ptr cl)
{
	detail::GaussianMatchResults ret;
	bool ok = m_model.match(ret, *cl, kf_pose(), particle_swarm());
	if (!ok) {
		std::cerr << "  [RioGaussian] Warning: Failed to improve score" << std::endl;
		return false;
	}

	auto L = ret.score - m_basemahal;
	std::cout << "RioGaussian scanmatch loss " << L << std::endl;
	if (L >= m_match_thresh) {
		std::cerr << "  [RioGaussian] Warning: Not converged" << std::endl;
		return false;
	}

	update_scanmatch(ret.pose);
	return true;
}

bool RioGaussian::process_keyframe(RadarCloud::Ptr cl)
{
	detail::GaussianFitParams p;

	p.num_gaussians = (cl->size() + cl->size()/2) / m_gaussian_sz;
	p.num_threads = num_threads();
	//p.verbose = true;

	std::cout << "RioGaussian: modeling " << p.num_gaussians << " gaussians with " << p.num_threads << " threads" << std::endl;

	m_model.fit(*cl, p);

	detail::GaussianMatchResults ret;
	bool ok = m_model.match(ret, *cl);
	m_basemahal = ret.score;
	std::cout << "RioGaussian: baseline score is " << m_basemahal << std::endl;
	if (ok) {
		std::cout << "RioGaussian: transforming model to suit better score" << std::endl;
		m_model.transform(ret.pose.inverse());
	}

	return true;
}

}
