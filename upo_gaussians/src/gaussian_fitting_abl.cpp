#include <upo_gaussians/gaussian_model.hpp>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/util/normal_estimation.hpp>

using upo_gaussians::AnyCloudIn;

namespace {
	struct AblModel {
		AnyCloudIn points;
		std::vector<Eigen::Matrix4d> covs;

		AblModel(AnyCloudIn in) : points{in}, covs{(size_t)in.cols()} {
			small_gicp::UnsafeKdTree<AblModel> kdtree{*this, small_gicp::KdTreeBuilder()};
			small_gicp::estimate_covariances(*this, kdtree, 10);
		}
	};
}

namespace small_gicp::traits {
	template <>
	struct Traits<AblModel> {
		static size_t size(AblModel const& self) { return self.points.cols(); }
		static auto point(AblModel const& self, size_t i) { return self.points.col(i).segment<4>(0).cast<double>(); }
		static auto const& cov(AblModel const& self, size_t i) { return self.covs[i]; }
		static void resize(AblModel& self, size_t n) { self.covs.resize(n); }
		static void set_cov(AblModel& self, size_t i, Eigen::Matrix4d const& cov) { self.covs[i] = cov; }
	};
}

namespace upo_gaussians {

void GaussianModel::fit_ablation(AnyCloudIn cl)
{
	AblModel m{cl};
	small_gicp::GaussianVoxelMap vm{2.0};
	vm.insert(m);

	size_t numg = vm.size();
	centers.resize(Eigen::NoChange, numg);
	log_scales.resize(Eigen::NoChange, numg);
	quats.resize(numg);

	for (size_t i = 0; i < numg; i ++) {
		auto& voxel = vm.flat_voxels[i]->second;
		centers.col(i) = voxel.mean.segment<3>(0);
		auto svd = voxel.cov.block<3,3>(0, 0).jacobiSvd(Eigen::ComputeFullU);
		log_scales.col(i) = (0.5*svd.singularValues().array().log()).max(std::log(0.05)).matrix();
		quats(i) = svd.matrixU().transpose() / svd.matrixU().determinant();
	}
}

}
