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
	MatchOut* out_grid_matchout
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

__global__ void icgSumReducePxy_Impl(
	uint32_t N_points,
	int32_t const* in_matchups,
	Vecf<4> const* in_points,
	Vecf<4> const* in_g_centers,
	Vecf<4> const* in_g_invscale,
	Quatf const* in_g_invrot,
	symposmat3* out_grid_mats,
	MultiVecf<4,2>* out_grid_xy
)
{
	auto block_idx = blockIdx.x + gridDim.x * blockIdx.y;
	auto transform_idx = blockIdx.y;
	auto point_idx = threadIdx.x + blockDim.x * blockIdx.x;
	auto lane_idx = threadIdx.x & (CUDA_WARP_SIZE - 1);
	auto warp_idx = threadIdx.x / CUDA_WARP_SIZE;

	// Fetch the index of the Gaussian associated to this thread
	int32_t g_idx = -1;
	if (point_idx < N_points) {
		g_idx = in_matchups[point_idx + N_points*transform_idx];
	}

	// Compute this thread's matrix
	symposmat3 P;
	MultiVecf<3,2> xy = MultiVecf<3,2>::Zero();
	if (g_idx >= 0) {
		Matf<3> M = in_g_invscale[g_idx].segment<3>(0).asDiagonal() * in_g_invrot[g_idx].toRotationMatrix();
		P = Matf<3>(M.transpose() * M);
		xy.col(0) = M.transpose() * (M * in_points[point_idx].segment<3>(0));
		xy.col(1) = M.transpose() * (M * in_g_centers[g_idx].segment<3>(0));
	}

	// Sum-reduce this warp's matrix
	for (unsigned w = CUDA_WARP_SIZE/2; w; w >>= 1) {
		for (unsigned i = 0; i < 2; i ++) {
			for (unsigned j = 0; j < 3; j ++) {
				P(i,j) += __shfl_xor_sync(UINT32_MAX, P(i,j), w);
				xy(j,i) += __shfl_xor_sync(UINT32_MAX, xy(j,i), w);
			}
		}
	}

	// Store each warp's matrix into sharedmem
	__shared__ Wrapper<decltype(P)>  block_P [UPO_CUDA_WARPS_PER_BLOCK];
	__shared__ Wrapper<decltype(xy)> block_xy[UPO_CUDA_WARPS_PER_BLOCK];
	if (lane_idx == 0) {
		*block_P[warp_idx] = P;
		*block_xy[warp_idx] = xy;
	}

	// Sum-reduce this thread block's matrix
	for (unsigned w = UPO_CUDA_WARPS_PER_BLOCK/2; w; w >>= 1) {
		__syncthreads();
		if (lane_idx == 0 && warp_idx < w) {
			*block_P[warp_idx] += *block_P[warp_idx+w];
			*block_xy[warp_idx] += *block_xy[warp_idx+w];
		}
	}

	// Store this thread block's reduced matrix
	if (lane_idx == 0 && warp_idx == 0) {
		out_grid_mats[block_idx] = *block_P[0];
		out_grid_xy[block_idx].block(0,0,3,2) = *block_xy[0];
	}
}

__global__ void icgSumReduceRotOpt_Impl(
	uint32_t N_points,
	int32_t const* in_matchups,
	MultiVecf<4,2> const* in_xy0,
	Quatf const* in_T_rot,
	Vecf<4> const* in_points,
	Vecf<4> const* in_g_centers,
	Vecf<4> const* in_g_invscale,
	Quatf const* in_g_invrot,
	Matf<12,3>* out_grid_M
)
{
	auto block_idx = blockIdx.x + gridDim.x * blockIdx.y;
	auto transform_idx = blockIdx.y;
	auto point_idx = threadIdx.x + blockDim.x * blockIdx.x;
	auto lane_idx = threadIdx.x & (CUDA_WARP_SIZE - 1);
	auto warp_idx = threadIdx.x / CUDA_WARP_SIZE;

	Vecf<3> x0 = in_xy0[transform_idx].col(0).segment<3>(0);
	Vecf<3> y0 = in_xy0[transform_idx].col(1).segment<3>(0);
	Quatf    q = in_T_rot[transform_idx];

	// Fetch the index of the Gaussian associated to this thread
	int32_t g_idx = -1;
	if (point_idx < N_points) {
		g_idx = in_matchups[point_idx + N_points*transform_idx];
	}

	Matf<12,3> M = Matf<12,3>::Zero();
	if (g_idx >= 0) {
		Matf<3> SR = in_g_invscale[g_idx].segment<3>(0).asDiagonal() * in_g_invrot[g_idx].toRotationMatrix();
		Matf<3> P  = SR.transpose()*SR;
		Vecf<3> u  = in_points[point_idx].segment<3>(0) - x0;
		Vecf<3> Ru = q*u;
		Vecf<3> v  = in_g_centers[g_idx].segment<3>(0) - y0;
		auto    U  = Ru*u.transpose();

		M.block(0,0,3,3) = -P*(Ru - v)*u.transpose();
		M.block(3,0,3,3) = P*skewsym(-U.col(0));
		M.block(6,0,3,3) = P*skewsym(-U.col(1));
		M.block(9,0,3,3) = P*skewsym(-U.col(2));
	}

	// Sum-reduce this warp's matrix
	for (unsigned w = CUDA_WARP_SIZE/2; w; w >>= 1) {
		for (unsigned i = 0; i < 12; i ++) {
			for (unsigned j = 0; j < 3; j ++) {
				M(i,j) += __shfl_xor_sync(UINT32_MAX, M(i,j), w);
			}
		}
	}

	// Store each warp's matrix into sharedmem
	__shared__ Wrapper<decltype(M)> block_M[UPO_CUDA_WARPS_PER_BLOCK];
	if (lane_idx == 0) {
		*block_M[warp_idx] = M;
	}

	// Sum-reduce this thread block's matrix
	for (unsigned w = UPO_CUDA_WARPS_PER_BLOCK/2; w; w >>= 1) {
		__syncthreads();
		if (lane_idx == 0 && warp_idx < w) {
			*block_M[warp_idx] += *block_M[warp_idx+w];
		}
	}

	// Store this thread block's reduced matrix
	if (lane_idx == 0 && warp_idx == 0) {
		out_grid_M[block_idx] = *block_M[0];
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

void IcgContext::cuda_sumReducePxy()
{
	icgSumReducePxy_Impl LAUNCH_ARGS(
		m_numPoints,
		m_matchups,
		m_points,
		m_g_centers,
		m_g_invscale,
		m_g_invrot,
		sr_pmat(),
		sr_xy0()
	);
}

void IcgContext::cuda_sumReduceRotOpt()
{
	icgSumReduceRotOpt_Impl LAUNCH_ARGS(
		m_numPoints,
		m_matchups,
		m_xy0,
		m_T_rot,
		m_points,
		m_g_centers,
		m_g_invscale,
		m_g_invrot,
		sr_rotopt()
	);
}

}
