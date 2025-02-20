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

	template <typename T>
	static constexpr T eps2 = T{1e-8};

	// https://github.com/strasdat/Sophus/blob/main/sophus/so3.hpp#L716-L752
	template <typename Derived>
	static inline auto so3_exp(Eigen::MatrixBase<Derived> const& v)
	{
		using T = typename Eigen::MatrixBase<Derived>::Scalar;
		static_assert(
			Eigen::MatrixBase<Derived>::RowsAtCompileTime==3 && Eigen::MatrixBase<Derived>::ColsAtCompileTime==1,
			"Bad dimensions"
		);

		T theta2 = v.squaredNorm();
		T im, re;
		if (theta2 < eps2<T>) {
			T theta4 = theta2*theta2;
			im = T{0.5} - T{1/48.0}*theta2 + T{1/3840.0}*theta4;
			re = T{1.0} - T{1/ 8.0}*theta2 + T{1/ 384.0}*theta4;
		} else {
			T theta = sqrt(theta2);
			T halftheta = T{0.5}*theta;
			im = sin(halftheta)/theta;
			re = cos(halftheta);
		}

		return Eigen::Quaternion<T>(re, im*v(0), im*v(1), im*v(2));
	}

	// https://github.com/pettni/smooth/blob/master/include/smooth/detail/so3.hpp#L124-L137
	template <typename T>
	static inline Vec<3,T> so3_log(Eigen::Quaternion<T> const& q_)
	{
		T xyz2 = q_.vec().squaredNorm();

		return T{2}*[&]() -> T {
			if (xyz2 < eps2<T>) {
				// https://www.wolframalpha.com/input/?i=series+atan%28y%2Fx%29+%2F+y+at+y%3D0
				return T{1}/q_.w() - xyz2/(T{3}*q_.w()*q_.w()*q_.w());
			} else {
				T xyz = sqrt(xyz2);
				return atan2(xyz, q_.w())/xyz;
			}
		}()*q_.vec();
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
