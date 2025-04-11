#include "gaussian_scan_matching.hpp"

#define LAUNCH_ARGS <<<dim3(m_numBlocks, m_numParticles), UPO_CUDA_NUM_THREADS>>>

namespace upo_gaussians::detail {

namespace {

__global__ void icgMatchupP2G_Impl(
	uint32_t N_points,
	uint32_t N_gaussians,
	float max_sqmahal,
	Vecf<4> const* in_T_tran,
	Quatf const* in_T_rot,
	Vecf<4> const* in_points,
	Vecf<4> const* in_g_centers,
	Vecf<4> const* in_g_invscale,
	Quatf const* in_g_invrot,
	int32_t* out_matchups,
	MatchOutf* out_grid_matchout
)
{
	auto block_idx = blockIdx.x + gridDim.x * blockIdx.y;
	auto transform_idx = blockIdx.y;
	auto point_idx = threadIdx.x + blockDim.x * blockIdx.x;
	auto lane_idx = threadIdx.x & (CUDA_WARP_SIZE - 1);
	auto warp_idx = threadIdx.x / CUDA_WARP_SIZE;

	bool is_active = point_idx < N_points;
	int32_t best_idx = -1;
	float best_sqmahal = is_active ? max_sqmahal : 0.0f;

	Vecf<3> point;
	if (is_active) {
		point = in_T_tran[transform_idx].segment<3>(0) + in_T_rot[transform_idx]*in_points[point_idx].segment<3>(0);
	}

	for (uint32_t base = 0; base < N_gaussians; base += CUDA_WARP_SIZE) {
		unsigned maxfetch = capToWarpSize(N_gaussians - base);

		Vecf<3> warp_g_center;
		Vecf<3> warp_g_invscale;
		Quatf   warp_g_invrot;
		if (lane_idx < maxfetch) {
			warp_g_center   = in_g_centers[base+lane_idx].segment<3>(0);
			warp_g_invscale = in_g_invscale[base+lane_idx].segment<3>(0);
			warp_g_invrot   = in_g_invrot[base+lane_idx];
		}

		for (unsigned i = 0; i < maxfetch; i ++) {
			Vecf<3> g_center;
			Vecf<3> g_invscale;
			Quatf   g_invrot;
			for (unsigned row = 0; row < 3; row ++) {
				g_center(row) = __shfl_sync(UINT32_MAX, warp_g_center(row), i);
				g_invscale(row) = __shfl_sync(UINT32_MAX, warp_g_invscale(row), i);
				g_invrot.coeffs()(row) = __shfl_sync(UINT32_MAX, warp_g_invrot.coeffs()(row), i);
			}
			g_invrot.coeffs()(3) = __shfl_sync(UINT32_MAX, warp_g_invrot.coeffs()(3), i);

			if (!is_active) {
				continue;
			}

			int32_t curidx = base+i;
			auto vector = g_invscale.cwiseProduct(g_invrot*(point - g_center));
			float sqmahal = vector.squaredNorm();

			if (sqmahal < best_sqmahal) {
				best_idx = curidx;
				best_sqmahal = sqmahal;
			}
		}
	}

	if (is_active) {
		out_matchups[point_idx + N_points*transform_idx] = best_idx;
	}

	// Sum-reduce for this warp
	uint32_t matched_mask = __ballot_sync(UINT32_MAX, best_idx >= 0);
	for (unsigned w = CUDA_WARP_SIZE/2; w; w >>= 1) {
		best_sqmahal += __shfl_xor_sync(UINT32_MAX, best_sqmahal, w);
	}

	// Prepare the thread block sum-reduce
	__shared__ float block_sqmahal[UPO_CUDA_WARPS_PER_BLOCK];
	__shared__ int32_t block_matches[UPO_CUDA_WARPS_PER_BLOCK];
	if (lane_idx == 0) {
		block_sqmahal[warp_idx] = best_sqmahal;
		block_matches[warp_idx] = __builtin_popcount(matched_mask);
	}

	// Sum-reduce for this thread block
	for (unsigned w = UPO_CUDA_WARPS_PER_BLOCK/2; w; w >>= 1) {
		__syncthreads();
		if (lane_idx == 0 && warp_idx < w) {
			block_sqmahal[warp_idx] += block_sqmahal[warp_idx+w];
			block_matches[warp_idx] += block_matches[warp_idx+w];
		}
	}

	// Store the reduced value
	if (lane_idx == 0 && warp_idx == 0) {
		auto& out = out_grid_matchout[block_idx];
		out.sqmahal = block_sqmahal[0];
		out.matches = block_matches[0];
	}
}

}

void IcgContext::cuda_matchupP2G(float max_mahal)
{
	icgMatchupP2G_Impl LAUNCH_ARGS(
		m_numPoints,
		m_numGaussians,
		max_mahal*max_mahal,
		m_T_tran,
		m_T_rot,
		m_points,
		m_g_centers,
		m_g_invscale,
		m_g_invrot,
		m_matchups,
		sr_matchOut()
	);
}

}

