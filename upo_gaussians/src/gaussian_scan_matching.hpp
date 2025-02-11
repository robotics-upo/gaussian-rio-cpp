#pragma once
#include <upo_gaussians/gaussian_model.hpp>
#include "cuda_helper.hpp"

namespace upo_gaussians::detail {

struct MatchOut {
	float sqmahal;
	uint32_t matches;
};

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

	GpuArray<Vecf<4>> m_points;
	GpuArray<Vecf<4>> m_g_centers;
	GpuArray<Vecf<4>> m_g_invscale;
	GpuArray<Quatf>   m_g_invrot;
	GpuArray<Vecf<4>> m_T_tran;
	GpuArray<Quatf>   m_T_rot;

	union SRTemp {
		MatchOut   match_out;
		symposmat3 pmat;
		Matf<12,3> rotopt;
	};

	// Temporary buffers (GPU/CPU shared memory)
	GpuArray<int32_t> m_matchups;
	GpuArray<MultiVecf<4,2>> m_xy0;
	GpuArray<SRTemp> m_sum_reduce_temp;
	GpuArray<MultiVecf<4,2>> m_sum_reduce_temp2;

	auto* sr_matchOut() { return &m_sum_reduce_temp[0].match_out; }
	auto* sr_pmat()     { return &m_sum_reduce_temp[0].pmat;      }
	auto* sr_rotopt()   { return &m_sum_reduce_temp[0].rotopt;    }
	auto* sr_xy0()      { return &m_sum_reduce_temp2[0];          }

	// CUDA kernels
	void cuda_matchupP2G(float max_mahal);
	void cuda_sumReducePxy();
	void cuda_sumReduceRotOpt();

public:
	IcgContext(
		AnyCloudIn cl,
		GaussianModel const& model,
		Pose const& init_pose,
		PoseArray const& particles
	);

	Pose particle(size_t i) const {
		return make_pose(m_T_rot[i].cast<double>(), m_T_tran[i].segment<3>(0).cast<double>())*m_initPose;
	}

	std::pair<size_t,double> matchup(float max_mahal);
	void iteration();
};

}
