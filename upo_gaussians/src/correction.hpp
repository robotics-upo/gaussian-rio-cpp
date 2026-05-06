#pragma once
#include <upo_gaussians/types.hpp>

namespace upo_gaussians::detail {

template <typename Scalar>
UPO_HOST_DEVICE static inline Scalar sech2(Scalar input)
{
	Scalar sech = Scalar{1.0}/std::cosh(input);
	return sech*sech;
}

template <typename Scalar>
UPO_HOST_DEVICE static inline Vec<3,Scalar> cauchy_loss(Scalar input, Scalar a = Scalar{1.0})
{
	Scalar b = a*a;
	Scalar c = Scalar{1.0}/b;

	Scalar sum = Scalar{1.0}+input*c;
	Scalar inv = Scalar{1.0}/sum;

	Vec<3,Scalar> ret;
	ret(0) = b*std::log(sum);
	ret(1) = std::max(std::numeric_limits<Scalar>::min(), inv);
	ret(2) = -c*(inv*inv);
	return ret;
}

template <typename Scalar>
UPO_HOST_DEVICE static inline Vec<3,Scalar> tanh_loss(Scalar input)
{
	Vec<3,Scalar> ret;
	ret(0) = std::tanh(input);
	ret(1) = sech2(input);
	ret(2) = -Scalar{2.0}*ret(0)*ret(1);
	return ret;
}

// Corrector using a robustifier function
// Based on https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/corrector.h, .cc
template <typename Scalar>
struct Corrector {
	Scalar m_sqrtRho1, m_resScale, m_alphaInput;

	static constexpr Scalar Zero = Scalar{0.0};
	static constexpr Scalar One  = Scalar{1.0};
	static constexpr Scalar Two  = Scalar{2.0};

	UPO_HOST_DEVICE explicit Corrector(Scalar input, Vec<3,Scalar> const& rho)
	{
		m_sqrtRho1 = std::sqrt(rho(1));

		if (input == Zero || rho(2) <= Zero) {
			m_resScale = m_sqrtRho1;
			m_alphaInput = Zero;
		} else {
			Scalar D = One + Two*input*rho(2)/rho(1);
			Scalar alpha = One - std::sqrt(D);
			m_resScale = m_sqrtRho1/(One - alpha);
			m_alphaInput = alpha/input;
		}
	}

	template <int Outputs>
	UPO_HOST_DEVICE void correct_residual(Vec<Outputs,Scalar>& residual)
	{
		residual *= m_resScale;
	}

	UPO_HOST_DEVICE void correct_residual(Scalar& residual)
	{
		residual *= m_resScale;
	}

	template <int Outputs, int Inputs>
	UPO_HOST_DEVICE void correct_jacobian(Mat<Outputs,Inputs,Scalar>& J, Vec<Outputs,Scalar> const& residual)
	{
		if (m_alphaInput == Zero) {
			J *= m_resScale;
			return;
		}

		for (int c = 0; c < Inputs; c ++) {
			Scalar rTj = Zero;

			for (int r = 0; r < Outputs; r ++) {
				rTj += J(r,c) * residual(r);
			}

			for (int r = 0; r < Outputs; r ++) {
				J(r,c) = m_sqrtRho1*(J(r,c) - m_alphaInput*residual(r)*rTj);
			}
		}
	}

	template <int Inputs>
	UPO_HOST_DEVICE void correct_gradient(Vec<Inputs,Scalar>& grad, Scalar residual)
	{
		if (m_alphaInput == Zero) {
			grad *= m_resScale;
			return;
		}

		for (int i = 0; i < Inputs; i ++) {
			Scalar rTj = grad(i)*residual;
			grad(i) = m_sqrtRho1*(grad(i) - m_alphaInput*residual*rTj);
		}
	}
};

}
