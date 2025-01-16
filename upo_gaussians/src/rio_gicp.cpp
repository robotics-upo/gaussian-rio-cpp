#include <upo_gaussians/rio_gicp.hpp>
#include <iostream>

#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>

namespace upo_gaussians {

class RioGicp::Model final {
	friend struct detail::RioGicpModelTraits;

	RioGicp const& m_parent;
	RadarCloud::Ptr m_cl;
	std::vector<Eigen::Matrix4d> m_covs;
	small_gicp::UnsafeKdTree<Model> m_kdtree;

public:
	Model(RioGicp const& parent, RadarCloud::Ptr&& cl) :
		m_parent{parent},
		m_cl{std::move(cl)},
		m_covs{m_cl->size()},
		m_kdtree{*this, small_gicp::KdTreeBuilderOMP(parent.num_threads())}
	{
		small_gicp::estimate_covariances_omp(*this, m_kdtree, parent.m_gicp_params.num_neighbors, parent.num_threads());
	}

	bool contains(RadarCloud const& cl) const { return m_cl.get() == &cl; }

	small_gicp::RegistrationResult align(Model const& cl) const
	{
		using namespace small_gicp;

		Registration<GICPFactor, ParallelReductionOMP> reg;
		reg.reduction.num_threads = m_parent.num_threads();
		reg.rejector.max_dist_sq = m_parent.m_gicp_params.max_corr_dist*m_parent.m_gicp_params.max_corr_dist;
		reg.criteria.rotation_eps = m_parent.m_gicp_params.rot_eps;
		reg.criteria.translation_eps = m_parent.m_gicp_params.tran_eps;
		reg.optimizer.max_iterations = m_parent.m_gicp_params.max_iters;
		reg.optimizer.verbose = false;

		return reg.align(*this, cl, m_kdtree, m_parent.kf_pose());
	}
};

namespace detail {

struct RioGicpModelTraits {
	using Points = RioGicp::Model;

	static size_t size(Points const& self) { return self.m_cl->size(); }

	static bool has_points(Points const& self) { return !self.m_cl->empty(); }
	static bool has_covs(Points const& self) { return !self.m_covs.empty(); }

	static auto point(Points const& self, size_t i) { return self.m_cl->operator[](i).getVector4fMap().cast<double>(); }
	static auto const& cov(Points const& self, size_t i) { return self.m_covs[i]; }

	static void resize(Points& self, size_t n) { self.m_covs.resize(n); }
	static void set_cov(Points& self, size_t i, Eigen::Matrix4d const& cov) { self.m_covs[i] = cov; }
};

}

RioGicp::RioGicp(
	Keyframer keyframer,
	InitParams const& p
) : RioBase{keyframer,p}, m_gicp_params{p.gicp_params}
{
}

RioGicp::~RioGicp() = default;

bool RioGicp::scan_matching(RadarCloud::Ptr cl)
{
	if (!m_model) {
		std::cerr << "  [GICP] Warning: No model to match" << std::endl;
		return false;
	}

	m_cached_model.reset();
	m_cached_model = std::make_unique<Model>(*this, std::move(cl));

	auto res = m_model->align(*m_cached_model);
	if (!res.converged) {
		std::cerr << "  [GICP] Warning: Not converged" << std::endl;
		return false;
	}

	update_scanmatch(res.T_target_source);
	return true;
}

bool RioGicp::process_keyframe(RadarCloud::Ptr cl)
{
	m_model.reset();

	if (m_cached_model && m_cached_model->contains(*cl)) {
		m_model = std::move(m_cached_model);
		m_cached_model.reset();
	} else {
		m_model = std::make_unique<Model>(*this, std::move(cl));
	}

	return true;
}

}

namespace small_gicp::traits {
	template <>
	struct Traits<upo_gaussians::RioGicp::Model> : upo_gaussians::detail::RioGicpModelTraits { };
}
