#pragma once
#include <upo_gaussians/gaussian_model.hpp>
#include "cuda_helper.hpp"

namespace upo_gaussians::detail {

namespace {

	template <typename Scalar>
	UPO_HOST_DEVICE Vec<6, Scalar> compress_sp3(Mat<3, 3, Scalar> const& m)
	{
		Vec<6, Scalar> ret;

		ret.template segment<3>(0) = m.row(0);
		ret.template segment<2>(3) = m.row(1).template segment<2>(1);
		ret(5) = m(2,2);

		return ret;
	}

	template <typename Scalar>
	UPO_HOST_DEVICE Mat<3, 3, Scalar> decompress_sp3(Vec<6, Scalar> const& v)
	{
		Mat<3, 3, Scalar> ret;

		ret.row(0) = v.template segment<3>(0);
		ret.col(0).template segment<2>(1) = v.template segment<2>(1);
		ret.row(1).template segment<2>(1) = v.template segment<2>(3);
		ret(2,1) = v(4);
		ret(2,2) = v(5);

		return ret;
	}

}

struct FitLoss {
	// Value
	float value = 0.0f;

	// Gradient (9 coefficients)
	Vecf<9> grad = Vecf<9>::Zero();

	// Compressed Hessian (42 coefficients)
	Matf<3> H_JcTJs = Matf<3>::Zero();
	Matf<3> H_JcTJq = Matf<3>::Zero();
	Matf<3> H_JsTJq = Matf<3>::Zero();
	Vecf<6> H_JcTJc = Vecf<6>::Zero();
	Vecf<6> H_JqTJq = Vecf<6>::Zero();
	Vecf<3> H_JsTJs = Vecf<3>::Zero();

	UPO_HOST_DEVICE FitLoss operator+=(FitLoss const& rhs) {
		value   += rhs.value;
		grad    += rhs.grad;
		H_JcTJs += rhs.H_JcTJs;
		H_JcTJq += rhs.H_JcTJq;
		H_JsTJq += rhs.H_JsTJq;
		H_JcTJc += rhs.H_JcTJc;
		H_JqTJq += rhs.H_JqTJq;
		H_JsTJs += rhs.H_JsTJs;

		return *this;
	}

	UPO_HOST_DEVICE Matf<9> hessian() const {
		Matf<9> ret;

		ret.block(0,0,3,3) = decompress_sp3(H_JcTJc);
		ret.block(0,3,3,3) = H_JcTJs;
		ret.block(0,6,3,3) = H_JcTJq;
		ret.block(3,0,3,3) = H_JcTJs.transpose();
		ret.block(3,3,3,3) = H_JsTJs.asDiagonal();
		ret.block(3,6,3,3) = H_JsTJq;
		ret.block(6,0,3,3) = H_JcTJq.transpose();
		ret.block(6,3,3,3) = H_JsTJq.transpose();
		ret.block(6,6,3,3) = decompress_sp3(H_JqTJq);

		return ret;
	}

};

template <unsigned N, typename Scalar = float>
struct Adam {
	unsigned epoch = 0;
	Vec<N, Scalar> m1 = Vec<N, Scalar>::Zero();
	Vec<N, Scalar> m2 = Vec<N, Scalar>::Zero();

	UPO_HOST_DEVICE Vec<N,Scalar> update(
		Vec<N,Scalar> const& grad,
		Scalar lr = Scalar{0.05},
		Scalar beta1 = Scalar{0.9},
		Scalar beta2 = Scalar{0.999},
		Scalar eps = Scalar{1.0e-15}
	) {
		static constexpr Scalar One = Scalar{1};

		m1 = beta1*m1 + (One-beta1)*grad;
		m2 = beta2*m2 + (One-beta2)*grad.array().square().matrix();

		epoch++;
		Scalar f1 = (One - std::pow(beta1, epoch));
		Scalar f2 = (One - std::pow(beta2, epoch));

		Vec<N,Scalar> m = m1 / f1;
		Vec<N,Scalar> v = m2 / f2;

		return -lr*(m.array()/(v.array().sqrt() + eps)).matrix();
	}
};

struct alignas(0x100) FitMup {
	union {
		uint32_t gaussian_id;
		uint32_t num_points;
	};

	FitLoss loss;
};

class FitContext {

	uint32_t m_numPoints;
	uint32_t m_capacity;
	uint32_t m_countedGaussians;
	float m_minLogScale;

	GpuArray<Vecf<4>>  m_points;
	GpuArray<FitMup>   m_matchups;

	GpuArray<uint32_t> m_enabled;
	GpuArray<Vecf<4>>  m_centers;
	GpuArray<Vecf<4>>  m_log_scales;
	GpuArray<Quatf>    m_rots;
	GpuArray<FitMup>   m_aggregates;

	std::vector<Adam<9>> m_adams;

	void cuda_launchMatchup();
	void cuda_launchAggregate();

	bool is_active(uint32_t gidx) const {
		return m_enabled[gidx/32] & (1U << (gidx&0x1f));
	}

	void set_active(uint32_t gidx, bool active) {
		uint32_t& st = m_enabled[gidx/32];
		uint32_t val = 1U << (gidx&0x1f);

		if (active) {
			st |= val;
		} else {
			st &= ~val;
		}
	}

public:
	FitContext(
		AnyCloudIn cl,
		VecArray<3> const& init_centers,
		uint32_t max_gaussians,
		double min_size = 0.05
	);

	constexpr size_t num_active_gaussians() const { return m_countedGaussians; }

	float matchup_and_loss();
	void iteration();

	void output(GaussianModel& gm) const;

};

}
