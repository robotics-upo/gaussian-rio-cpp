#include "gaussian_fitting.hpp"
#include <upo_gaussians/bisecting_kmeans.hpp>

#include <chrono>

#define WANT_SECOND_ORDER

namespace upo_gaussians {

namespace detail {

FitContext::FitContext(
	AnyCloudIn cl,
	VecArray<3> const& init_centers,
	uint32_t max_gaussians,
	double min_size
) :
	m_numPoints{(uint32_t)cl.cols()},
	m_capacity{(max_gaussians + CUDA_WARP_SIZE - 1) &~ (CUDA_WARP_SIZE - 1)},
	m_countedGaussians{(uint32_t)init_centers.cols()},
	m_minLogScale{(float)std::log(min_size)},
	m_adams{m_capacity}
{
	m_points.reserve(m_numPoints);
	m_matchups.reserve(m_numPoints);

	m_enabled.reserve(m_capacity/32);
	m_centers.reserve(m_capacity);
	m_log_scales.reserve(m_capacity);
	m_rots.reserve(m_capacity);
	m_aggregates.reserve(m_capacity);

	for (uint32_t i = 0; i < m_numPoints; i ++) {
		m_points[i].segment<3>(0) = cl.col(i).segment<3>(0);
	}

	memset(m_enabled, 0, m_capacity/8);

	for (unsigned i = 0; i < m_countedGaussians; i ++) {
		set_active(i, true);

		m_centers[i].segment<3>(0) = init_centers.col(i).cast<float>();
		m_centers[i](3) = 0.0f;
		m_log_scales[i] = m_minLogScale*Vecf<4>::Ones();
		m_rots[i] = Quatf::Identity();
	}
}

float FitContext::matchup_and_loss()
{
	//auto tref = std::chrono::high_resolution_clock::now();
	cuda_launchMatchup();
	cuda_launchAggregate();
	cudaDeviceSynchronize();
	//std::chrono::duration<double, std::milli> elapsed = std::chrono::high_resolution_clock::now() - tref;

	uint32_t total_g = 0;
	double total_loss = 0.0;

	for (unsigned i = 0; i < m_capacity; i ++) {
		if (!is_active(i)) {
			continue;
		}

		if (m_aggregates[i].num_points < 4) {
			// Prune this Gaussian
			set_active(i, false);
			continue;
		}

		total_g ++;
		total_loss += m_aggregates[i].loss.value;
	}

	m_countedGaussians = total_g;
	if (total_g != 0) {
		total_loss /= total_g;
	}

	return total_loss;
}

void FitContext::iteration()
{
	float total_lambda = 0.0f;
	float total_change = 0.0f;

	for (unsigned i = 0; i < m_capacity; i ++) {
		auto const& agg = m_aggregates[i];
		if (!is_active(i)) {
			continue;
		}

		Vecf<9> loss_grad = agg.loss.grad;

		/*
		printf("  Gaussian %u has %u points with loss %f\n", i, agg.num_points, agg.loss.value);
		std::cout << "    pos [" << m_centers[i].segment<3>(0).transpose() << "]" << std::endl;
		std::cout << "    grad [" << loss_grad.transpose() << "]" << std::endl;
		//std::cout << "    hessian [" << loss_H << "]" << std::endl;
		*/

#ifdef WANT_SECOND_ORDER

		float lambda = 1.0f/1024;
		Matf<9> loss_H = agg.loss.hessian();
		Vecf<9> change;

		for (;;) {
			change = (loss_H + lambda*Matf<9>::Identity()).ldlt().solve(-loss_grad);
			if (change.norm() <= 1.0f) {
				break;
			}

			lambda *= 2.0f;
		}

#else

		float lambda = 0.0f;
		float lr = 0.05f;
		Adam<9>& adam = m_adams[i];
		Vecf<9> change = adam.update(loss_grad, lr);

#endif

		total_lambda += lambda;
		total_change += change.norm();

		// Update parameters
		m_centers[i].segment<3>(0) += change.segment<3>(0);
		m_log_scales[i].segment<3>(0) += change.segment<3>(3);
		m_rots[i] = so3_exp(change.segment<3>(6))*m_rots[i];

		// Renormalize
		m_log_scales[i] = m_log_scales[i].cwiseMax(m_minLogScale);
		m_rots[i].normalize();
	}

	total_lambda /= m_countedGaussians;
	total_change /= m_countedGaussians;
	printf("FitContext::iteration(): mean lambda = %f change = %f\n", total_lambda, total_change);
}

void FitContext::output(GaussianModel& gm) const
{
	gm.centers.resize(Eigen::NoChange, m_countedGaussians);
	gm.log_scales.resize(Eigen::NoChange, m_countedGaussians);
	gm.quats.resize(m_countedGaussians);

	unsigned gidx = 0;
	for (unsigned i = 0; i < m_capacity; i ++) {
		if (!is_active(i)) {
			continue;
		}

		gm.centers.col(gidx) = m_centers[i].segment<3>(0).cast<double>();
		gm.log_scales.col(gidx) = m_log_scales[i].segment<3>(0).cast<double>();
		gm.quats(gidx) = m_rots[i].cast<double>();
		gidx++;
	}
}

}

void GaussianModel::fit(AnyCloudIn cl, FitParams const& p)
{
	std::mt19937 g{3135134162};
	BisectingKMeans kmeans{cl, g, p.max_gaussians, p.points_per_g > 2 ? p.points_per_g : 2};

	kmeans.get_centers(centers);

	printf("We have %f points per Gaussian\n", (double)cl.cols() / centers.cols());

	std::cout << "We have " << centers.cols() << " Gaussian centers" << std::endl;

	detail::FitContext ctx(cl, centers, centers.cols());

	float loss = ctx.matchup_and_loss();
	printf("Initial loss: %f over %zu Gaussians\n", loss, ctx.num_active_gaussians());

	for (unsigned epoch = 0; epoch < 20; epoch ++) {
		ctx.iteration();
		loss = ctx.matchup_and_loss();

		printf("Epoch %u: loss = %f over %zu Gaussians\n", epoch, loss, ctx.num_active_gaussians());

		/*if (loss < 0.0f) {
			break;
		}*/
	}

	//std::exit(0);
	//std::abort();

	ctx.output(*this);

}

}
