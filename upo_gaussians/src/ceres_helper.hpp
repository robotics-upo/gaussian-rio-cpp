#pragma once
#include <ceres/ceres.h>

#if ((CERES_VERSION_MAJOR<<8) | CERES_VERSION_MINOR) < 0x201
#define EigenQuaternionManifold EigenQuaternionParameterization
#define manifold_ownership local_parameterization_ownership
#endif

namespace upo_gaussians::detail {

template <typename CostFunctor, int kNumResiduals, int... Ns>
class GroupedAutoDiffCostFunction : public ceres::CostFunction {
	std::unique_ptr<CostFunctor> functor_;
	ceres::Ownership ownership_;
	size_t num_groups_ = 0;

	using ParamDims = typename ceres::SizedCostFunction<kNumResiduals, Ns...>::ParameterDims;

public:
	size_t cur_group_id = 0;

	explicit GroupedAutoDiffCostFunction(CostFunctor* functor, ceres::Ownership ownership = ceres::TAKE_OWNERSHIP) :
		functor_{functor}, ownership_{ownership}
	{
		set_num_residuals(kNumResiduals);
	}

	explicit GroupedAutoDiffCostFunction(GroupedAutoDiffCostFunction&& other) :
		functor_{std::move(other.functor_)}, ownership_{other.ownership_} {}

	virtual ~GroupedAutoDiffCostFunction()
	{
		if (ownership_ == ceres::DO_NOT_TAKE_OWNERSHIP) {
			functor_.release();
		}
	}

	void SetNumGroups(size_t num)
	{
		num_groups_ = num;

		auto& sizes = *mutable_parameter_block_sizes();
		sizes.clear();
		sizes.reserve(num*sizeof...(Ns));
		for (size_t i = 0; i < num; i ++) {
			(sizes.emplace_back(Ns), ...);
		}
	}

	bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
	{
		if (!jacobians) {
			return ceres::internal::VariadicEvaluate<ParamDims>(
				*functor_,
				parameters + cur_group_id*ParamDims::kNumParameterBlocks,
				residuals
			);
		}

		// Fill out unused jacobians
		for (size_t i = 0; i < num_groups_; i ++) {
			if (i == cur_group_id) {
				continue;
			}

			double** group_jacobians = jacobians + i*ParamDims::kNumParameterBlocks;
			for (size_t j = 0; j < ParamDims::kNumParameterBlocks; j ++) {
				if (group_jacobians[j]) {
					std::fill_n(group_jacobians[j], ParamDims::GetDim(j)*num_residuals(), 0.0);
				}
			}
		}

		return ceres::internal::AutoDifferentiate<kNumResiduals, ParamDims>(
			*functor_,
			parameters + cur_group_id*ParamDims::kNumParameterBlocks,
			num_residuals(),
			residuals,
			jacobians + cur_group_id*ParamDims::kNumParameterBlocks
		);
	}
};

}
