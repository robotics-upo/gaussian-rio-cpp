#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>

// C++
#include <vector>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#ifndef M_TAU
#define M_TAU (2*M_PI)
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

	using Quat = Eigen::Quaterniond;
	using Pose = Eigen::Isometry3d;

	static inline Mat<3> skewsym(Vec<3> const& v)
	{
		Mat<3> m; m <<
			+0,     -v.z(), +v.y(),
			+v.z(), +0,     -v.x(),
			-v.y(), +v.x(), +0;
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
		ret.rotate(rot);
		ret.translate(tran);
		return ret;
	}

	template <typename T>
	auto const& as_dense(Eigen::MatrixBase<T> const& x) { return x; }
	template <typename T>
	auto as_dense(Eigen::DiagonalWrapper<T> const& x) { return x.toDenseMatrix(); }
}
