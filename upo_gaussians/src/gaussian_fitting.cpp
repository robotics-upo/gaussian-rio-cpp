#include <upo_gaussians/gaussian_model.hpp>
#include <upo_gaussians/bisecting_kmeans.hpp>
#include <iostream>
#include <omp.h>

#include "ceres_helper.hpp"

namespace upo_gaussians {

namespace {

class GaussianModelMaker final : private ceres::EvaluationCallback {

	class PointCost final {
		detail::GroupedAutoDiffCostFunction<PointCost, 3, 3, 3, 4> m_base { this, ceres::DO_NOT_TAKE_OWNERSHIP };
		GaussianModelMaker& m_parent;

		size_t point_id() const { return this - m_parent.m_point_costs.data(); }

		template <typename T>
		auto point() const { return m_parent.point<T>(point_id()); }

	public:
		PointCost(GaussianModelMaker& parent) : m_parent{parent} { }
		PointCost(PointCost const&) = delete;
		PointCost(PointCost&&) = default;

		operator ceres::CostFunction*() { return &m_base; }

		size_t& gaussian_id()           { return m_base.cur_group_id; }
		size_t  gaussian_id()     const { return m_base.cur_group_id; }
		double  gaussian_weight() const { return m_parent.m_gaussian_weights(gaussian_id()); }
		void set_num_gaussians(size_t num) { m_base.SetNumGroups(num); }

		template <typename T>
		bool operator()(
			T const* in_g_center,
			T const* in_g_scale,
			T const* in_g_quat,
			T*       out_residual
		) const
		{
			Eigen::Map<Eigen::Vector<T,3>   const> g_center { in_g_center  };
			Eigen::Map<Eigen::Vector<T,3>   const> g_scale  { in_g_scale   };
			Eigen::Map<Eigen::Quaternion<T> const> g_quat   { in_g_quat    };
			Eigen::Map<Eigen::Vector<T,3>>         residual { out_residual };

			Eigen::Vector<T,3> pt = point<T>();
			residual = T{gaussian_weight()} * ((g_quat.conjugate()*(pt - g_center)).array()*(-g_scale).array().exp()).matrix();

			return true;
		}
	};

	class GaussianCost final {
		ceres::AutoDiffCostFunction<GaussianCost, 4, 3> m_base { this, ceres::DO_NOT_TAKE_OWNERSHIP };
		GaussianModelMaker& m_parent;

	public:
		GaussianCost(GaussianModelMaker& parent) : m_parent{parent} { }
		GaussianCost(GaussianCost const&) = delete;
		GaussianCost(GaussianCost&&) = default;

		operator ceres::CostFunction*() { return &m_base; }

		template <typename T>
		bool operator()(T const* in_g_scale, T* out_residual) const
		{
			Eigen::Map<Eigen::Vector<T,3> const> g_scale  { in_g_scale   };
			Eigen::Map<Eigen::Vector<T,3>>       residual { out_residual };

			residual = (0.5*g_scale).array().exp().matrix();
			out_residual[3] = exp(0.5*std::max(T{0.0}, g_scale.minCoeff() - T{m_parent.m_disc_size}));

			return true;
		}
	};

	ceres::Problem m_problem{ make_ceres_problem_options() };
	GaussianModel& m_model;
	AnyCloudIn m_cloud;
	std::vector<PointCost> m_point_costs;
	GaussianCost m_gaussian_cost{*this};
	Eigen::Vector<size_t, Eigen::Dynamic> m_group_sizes;
	Eigen::Vector<double, Eigen::Dynamic> m_gaussian_weights;
	ceres::EigenQuaternionManifold m_quat_manifold;
	ceres::CauchyLoss m_cauchy_loss{1.0};

	double m_scale_base;
	double m_disc_size;
	uint16_t m_num_threads;
	bool m_verbose;

	template <typename T>
	auto point(size_t id) const {
		return m_cloud.col(id).segment<3>(0).cast<T>();
	}

	ceres::Problem::Options make_ceres_problem_options()
	{
		ceres::Problem::Options opt;
		opt.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
		opt.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
		opt.manifold_ownership      = ceres::DO_NOT_TAKE_OWNERSHIP;
		opt.evaluation_callback = this;
		return opt;
	}

	virtual void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point)
	{
		if (!new_evaluation_point) {
			return;
		}

		m_group_sizes.fill(0);
		size_t* sizes = m_group_sizes.data();

		Eigen::Matrix<double, 3, Eigen::Dynamic> g_invnormscales =
			(-(m_model.log_scales.rowwise() - m_model.log_scales.colwise().mean())).array().exp().matrix();

		#pragma omp parallel for reduction(+:sizes[:m_group_sizes.size()]) num_threads(m_num_threads)
		for (Eigen::Index i = 0; i < m_cloud.cols(); i ++) {
			ssize_t best_g = -1;
			double best_sqdist = 0.0;

			Eigen::Vector3d pt = point<double>(i);

			for (size_t j = 0; j < (size_t)m_model.size(); j ++) {
				double cur_sqdist = g_invnormscales.col(j).cwiseProduct(m_model.quats(j).conjugate()*(pt - m_model.centers.col(j))).squaredNorm();

				if (best_g < 0 || cur_sqdist < best_sqdist) {
					best_g = j;
					best_sqdist = cur_sqdist;
				}
			}

			m_point_costs[i].gaussian_id() = best_g;
			++sizes[best_g];
		}

		m_gaussian_weights = (2.0*m_group_sizes.cast<double>()).array().rsqrt().matrix();

		if (m_verbose) {
			std::cout << "P2G assignments: [" << m_group_sizes.transpose() << "]" << std::endl;
		}
	}

public:

	using InitParams = detail::GaussianFitParams;

	GaussianModelMaker(GaussianModel& model, AnyCloudIn cl, InitParams const& p) :
		m_model{model},
		m_cloud{cl},
		m_scale_base{std::log(p.min_scale)},
		m_disc_size{std::log(p.disc_thickness)},
		m_num_threads{p.num_threads},
		m_verbose{p.verbose}
	{
		{
			std::mt19937 g{3135134162};
			BisectingKMeans{cl, g, p.num_gaussians}.get_centers(m_model.centers);
		}

		auto num_gaussians = m_model.size();
		m_model.log_scales.resize(Eigen::NoChange, num_gaussians);
		m_model.quats.resize(num_gaussians);

		m_model.log_scales.fill(std::log(p.initial_scale));
		m_model.quats.fill(Eigen::Quaterniond::Identity());

		m_point_costs.reserve(cl.cols());

		m_group_sizes.resize(num_gaussians);
		m_gaussian_weights.resize(num_gaussians);
		m_gaussian_weights.fill(0.0);

		std::vector<double*> point_cost_params;
		point_cost_params.reserve(3*num_gaussians);

		for (Eigen::Index i = 0; i < num_gaussians; i ++) {
			double* g_center = m_model.centers.col(i).data();
			double* g_scale  = m_model.log_scales.col(i).data();
			double* g_quat   = m_model.quats(i).coeffs().data();

			m_problem.AddParameterBlock(g_center, 3);
			m_problem.AddParameterBlock(g_scale,  3);
			m_problem.AddParameterBlock(g_quat,   4, &m_quat_manifold);

			for (int j = 0; j < 3; j ++) {
				m_problem.SetParameterLowerBound(g_scale, j, m_scale_base);
			}

			m_problem.AddResidualBlock(m_gaussian_cost, &m_cauchy_loss, g_scale);
			point_cost_params.emplace_back(g_center);
			point_cost_params.emplace_back(g_scale);
			point_cost_params.emplace_back(g_quat);
		}

		for (Eigen::Index i = 0; i < cl.cols(); i ++) {
			auto& cost = m_point_costs.emplace_back(*this);
			cost.set_num_gaussians(num_gaussians);
			m_problem.AddResidualBlock(cost, nullptr, point_cost_params);
		}
	}

	ceres::Solver::Summary solve()
	{
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		options.minimizer_progress_to_stdout = m_verbose;
		options.num_threads = m_num_threads;
		options.dynamic_sparsity = true;
		ceres::Solver::Summary summary;
		Solve(options, &m_problem, &summary);

		return summary;
	}

};

}

void GaussianModel::fit(AnyCloudIn cl, FitParams const& p)
{
	GaussianModelMaker maker{*this, cl, p};
	auto result = maker.solve();

	if (p.verbose) {
		std::cout << result.FullReport() << std::endl;
	}
}

}
