#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>

// C++
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#ifndef M_TAU
#define M_TAU (2*M_PI)
#endif

#ifdef __CUDACC__
#define UPO_HOST_DEVICE __host__ __device__
#else
#define UPO_HOST_DEVICE
#endif

namespace upo_gaussians {

	constexpr auto Dyn = Eigen::Dynamic;

	template <int rows, int cols = rows, typename T = double>
	using Mat = Eigen::Matrix<T, rows, cols, Eigen::RowMajor>;
	template <int rows, int cols = rows>
	using Matf = Mat<rows, cols, float>;
	using DynMatf = Eigen::Matrix<float,  Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using DynMatd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

	template <int elems, typename T = double>
	using Vec = Eigen::Vector<T, elems>;
	template <int elems>
	using Vecf = Vec<elems,float>;
	using DynVecf = Eigen::VectorXf;
	using DynVecd = Eigen::VectorXd;

	template <int rows, int cols, typename T = double>
	using MultiVec = Eigen::Matrix<T, rows, cols, Eigen::ColMajor>;
	template <int rows, int cols>
	using MultiVecf = MultiVec<rows, cols, float>;

	template <int rows, typename T = double>
	using VecArray = Eigen::Matrix<T, rows, Eigen::Dynamic, Eigen::ColMajor>;

	using AnyCloudIn  = Eigen::Ref<Eigen::MatrixXf const, Eigen::Aligned, Eigen::OuterStride<>> const&;
	using AnyCloudOut = Eigen::Ref<Eigen::MatrixXf,       Eigen::Aligned, Eigen::OuterStride<>>;

	using Quat = Eigen::Quaterniond;
	using Quatf = Eigen::Quaternionf;
	using Pose = Eigen::Isometry3d;

	using QuatArray = Eigen::Vector<Eigen::Quaterniond, Eigen::Dynamic>;
	using PoseArray = Eigen::Vector<Eigen::Isometry3d, Eigen::Dynamic>;

	template <typename Derived>
	UPO_HOST_DEVICE static inline auto skewsym(Eigen::MatrixBase<Derived> const& v)
	{
		using T = typename Eigen::MatrixBase<Derived>::Scalar;
		Mat<3,3,T> m; m <<
			T{0.0}, -v.z(), +v.y(),
			+v.z(), T{0.0}, -v.x(),
			-v.y(), +v.x(), T{0.0};
		return m;
	}

	static inline double sinc(double x)
	{
		x = fabs(x) + 1e-14;
		return sin(x) / x;
	}

	static inline Quat pure_quat_exp(Vec<3> const& v)
	{
		double norm = v.norm();
		double qw = cos(norm);
		auto qv = sinc(norm)*v;
		return Quat(qw, qv.x(), qv.y(), qv.z());
	}

	static inline Pose make_pose(Quat const& rot, Vec<3> const& tran)
	{
		Pose ret = Pose::Identity();
		ret.matrix().block(0,0,3,3) = rot.toRotationMatrix();
		ret.matrix().block(0,3,3,1) = tran;
		return ret;
	}

	template <typename T>
	auto const& as_dense(Eigen::MatrixBase<T> const& x) { return x; }
	template <typename T>
	auto as_dense(Eigen::DiagonalWrapper<T> const& x) { return x.toDenseMatrix(); }
}
