#pragma once
#include <upo_gaussians/gaussian_model.hpp>
#include "cuda_helper.hpp"

namespace upo_gaussians::detail {

template <typename T>
struct MatchOut {
	T sqmahal;
	uint32_t matches;
};

using MatchOutf = MatchOut<float>;
using MatchOutd = MatchOut<double>;

template <typename Scalar>
struct SPMat3 {
	using ReprType = Mat<2,4,Scalar>;
	using ExpType = Mat<3,3,Scalar>;

	ReprType m_repr = ReprType::Zero();

	UPO_HOST_DEVICE constexpr SPMat3() = default;

	UPO_HOST_DEVICE SPMat3(ExpType const& m) {
		m_repr.block(0,0,1,3) = m.diagonal().transpose();
		m_repr.block(1,0,1,2) = m.block(0,1,1,2);
		m_repr(1,2) = m(1,2);
	}

	template <typename Scalar2>
	UPO_HOST_DEVICE auto cast() const {
		SPMat3<Scalar2> ret;
		ret.m_repr = m_repr.template cast<Scalar2>();
		return ret;
	}

	UPO_HOST_DEVICE operator ExpType() const {
		ExpType ret = ExpType::Zero();
		ret.diagonal().transpose() = m_repr.block(0,0,1,3);
		ret.block(0,1,1,2) = ret.block(1,0,2,1).transpose() = m_repr.block(1,0,1,2);
		ret(1,2) = ret(2,1) = m_repr(1,2);
		return ret;
	}

	UPO_HOST_DEVICE auto& operator ()(size_t i, size_t j) {
		return m_repr(i,j);
	}

	UPO_HOST_DEVICE auto& operator ()(size_t i, size_t j) const {
		return m_repr(i,j);
	}

	UPO_HOST_DEVICE SPMat3& operator +=(SPMat3 const& rhs) {
		m_repr += rhs.m_repr;
		return *this;
	}
};

using symposmat3 = SPMat3<float>;

class IcgContext {
	Pose const& m_initPose;

	uint32_t m_numPoints;
	uint32_t m_numGaussians;
	uint32_t m_numParticles;
	uint32_t m_numBlocks;

	uint64_t m_convergedParticles = 0;

	GpuArray<Vecf<4>> m_points;
	GpuArray<Vecf<4>> m_g_centers;
	GpuArray<Vecf<4>> m_g_invscale;
	GpuArray<Quatf>   m_g_invrot;
	GpuArray<Vecf<4>> m_T_tran;
	GpuArray<Quatf>   m_T_rot;

	union SRTemp {
		MatchOutf  match_out;
	};

	// Temporary buffers (GPU/CPU shared memory)
	GpuArray<int32_t> m_matchups;
	GpuArray<SRTemp> m_sum_reduce_temp;

	auto* sr_matchOut() { return &m_sum_reduce_temp[0].match_out; }

	// CUDA kernels
	void cuda_matchupP2G(float max_mahal);
	void cuda_sumReducePxy();
	void cuda_sumReduceRotOpt();

	Pose particle_cur(size_t i) const {
		return make_pose(m_T_rot[i].cast<double>(), m_T_tran[i].segment<3>(0).cast<double>());
	}

public:
	IcgContext(
		AnyCloudIn cl,
		GaussianModel const& model,
		Pose const& init_pose,
		PoseArray const& particles
	);

	uint32_t num_particles() const {
		return m_numParticles;
	}

	uint32_t num_converged_particles() const {
		return __builtin_popcountll(m_convergedParticles);
	}

	Pose particle(size_t i) const {
		return particle_cur(i)*m_initPose;
	}

	std::pair<size_t,double> matchup(float max_mahal);
	void iteration(double min_change_rot, double min_change_tran);
};

}
