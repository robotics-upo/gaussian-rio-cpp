#include "gaussian_fitting.hpp"
#include "correction.hpp"

namespace upo_gaussians::detail {

namespace {

__global__ void fitMatchup_Impl(
	uint32_t N_points,
	uint32_t N_gaussians,
	Vecf<4> const* in_points,
	uint32_t const* in_g_mask,
	Vecf<4> const* in_g_centers,
	Vecf<4> const* in_g_log_scales,
	Quatf const* in_g_rots,
	FitMup* out_mups
)
{
	auto point_idx = threadIdx.x + blockDim.x * blockIdx.x;
	auto lane_idx = threadIdx.x & (CUDA_WARP_SIZE - 1);
	bool is_active = point_idx < N_points;

	int32_t best_g = -1;
	float best_sqdist = 0.0f;

	Vecf<3> point;
	if (is_active) {
		point = in_points[point_idx].segment<3>(0);
	}

	for (uint32_t base = 0; base < N_gaussians; base += CUDA_WARP_SIZE) {
		uint32_t active_mask = in_g_mask[base/CUDA_WARP_SIZE];
		if (!active_mask) {
			continue;
		}

		Vecf<3> warp_g_center = in_g_centers[base+lane_idx].segment<3>(0);

		for (unsigned i = 0; i < CUDA_WARP_SIZE; i ++) {
			Vecf<3> g_center;
			for (unsigned row = 0; row < 3; row ++) {
				g_center(row) = __shfl_sync(UINT32_MAX, warp_g_center(row), i);
			}

			if (!is_active || !(active_mask & (1U << i))) {
				continue;
			}

			int32_t curg = base+i;
			float sqdist = (point-g_center).squaredNorm();

			if (best_g < 0 || sqdist < best_sqdist) {
				best_g = curg;
				best_sqdist = sqdist;
			}
		}
	}

	if (!is_active) {
		return;
	}

	FitMup& mup = out_mups[point_idx];

	Vecf<3> g_center = in_g_centers[best_g].segment<3>(0);
	Vecf<3> g_log_scale = in_g_log_scales[best_g].segment<3>(0);
	Vecf<3> g_invscale = (-g_log_scale).array().exp().matrix();
	Quatf g_invquat = in_g_rots[best_g].conjugate();

	Vecf<3> residual = g_invscale.cwiseProduct(g_invquat*(point - g_center));

	// Compressed Jacobian
	Matf<3> Jc = g_invscale.asDiagonal()*(-g_invquat.toRotationMatrix());
	Vecf<3> Js = -residual;
	Matf<3> Jq = g_invscale.asDiagonal()*g_invquat.toRotationMatrix()*skewsym(point - g_center);

	mup.gaussian_id = best_g;

	mup.loss.value = residual.squaredNorm();

	mup.loss.grad.segment<3>(0) = Jc.transpose()*residual;
	mup.loss.grad.segment<3>(3) = Js.cwiseProduct(residual);
	mup.loss.grad.segment<3>(6) = Jq.transpose()*residual;

	mup.loss.H_JcTJs = Jc.transpose()*Js.asDiagonal();
	mup.loss.H_JcTJq = Jc.transpose()*Jq;
	mup.loss.H_JsTJq = Js.asDiagonal()*Jq;
	mup.loss.H_JcTJc = compress_sp3<float>(Jc.transpose()*Jc);
	mup.loss.H_JqTJq = compress_sp3<float>(Jq.transpose()*Jq);
	mup.loss.H_JsTJs = Js.array().square().matrix();
}

__global__ void fitAggregate_Impl(
	uint32_t N_points,
	uint32_t N_gaussians,
	FitMup const* in_mups,
	uint32_t const* in_g_mask,
	Vecf<4> const* in_g_log_scales,
	FitMup* out_agg
)
{
	auto gaussian_idx = threadIdx.x + blockDim.x * blockIdx.x;
	auto lane_idx = threadIdx.x & (CUDA_WARP_SIZE - 1);
	bool is_active = gaussian_idx < N_gaussians && (in_g_mask[gaussian_idx/CUDA_WARP_SIZE] & (1U << lane_idx));

	uint32_t pcount = 0;
	FitLoss loss;

	for (uint32_t base = 0; base < N_points; base += CUDA_WARP_SIZE) {
		unsigned maxfetch = capToWarpSize(N_points - base);

		uint32_t warp_gidx;
		if (lane_idx < maxfetch) {
			warp_gidx = in_mups[base+lane_idx].gaussian_id;
		}

		for (unsigned i = 0; i < maxfetch; i ++) {
			uint32_t p_gidx = __shfl_sync(UINT32_MAX, warp_gidx, i);

			if (!is_active || p_gidx != gaussian_idx) {
				continue;
			}

			FitMup const& mup = in_mups[base+i];
			pcount++;
			loss += mup.loss;
		}
	}

	if (!is_active) {
		return;
	}

	FitMup& agg = out_agg[gaussian_idx];
	agg.num_points = pcount;

	if (!pcount) {
		return;
	}

	float invpcount = 1.0f/pcount;
	Vecf<3> g_log_scale = in_g_log_scales[gaussian_idx].segment<3>(0);

	agg.loss.value = 0.5f*invpcount*loss.value + g_log_scale.array().sum();
	agg.loss.grad = invpcount*loss.grad;
	agg.loss.grad.segment<3>(3) += Vecf<3>{1.0f, 1.0f, 1.0f};
	agg.loss.H_JcTJs = invpcount*loss.H_JcTJs;
	agg.loss.H_JcTJq = invpcount*loss.H_JcTJq;
	agg.loss.H_JsTJq = invpcount*loss.H_JsTJq;
	agg.loss.H_JcTJc = invpcount*loss.H_JcTJc;
	agg.loss.H_JqTJq = invpcount*loss.H_JqTJq;
	agg.loss.H_JsTJs = invpcount*loss.H_JsTJs;
}

}

void FitContext::cuda_launchMatchup()
{
	uint32_t num_blocks = (m_numPoints + UPO_CUDA_NUM_THREADS - 1) / UPO_CUDA_NUM_THREADS;

	fitMatchup_Impl<<<num_blocks, UPO_CUDA_NUM_THREADS>>>(
		m_numPoints,
		m_capacity,
		m_points,
		m_enabled,
		m_centers,
		m_log_scales,
		m_rots,
		m_matchups
	);
}

void FitContext::cuda_launchAggregate()
{
	uint32_t num_blocks = (m_capacity + UPO_CUDA_NUM_THREADS - 1) / UPO_CUDA_NUM_THREADS;

	fitAggregate_Impl<<<num_blocks, UPO_CUDA_NUM_THREADS>>>(
		m_numPoints,
		m_capacity,
		m_matchups,
		m_enabled,
		m_log_scales,
		m_aggregates
	);
}

}

