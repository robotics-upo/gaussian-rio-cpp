#pragma once
#include "types.hpp"
#include "radar.hpp"

namespace upo_gaussians {

	namespace detail {

		struct GaussianFitParams {
			size_t   num_gaussians  = 150;
			double   initial_scale  = 1.0;
			double   min_scale      = 0.05;
			double   disc_thickness = 0.15;
			uint16_t num_threads    = 4;
			bool     verbose        = false;
		};

		struct GaussianMatchParams {
			uint32_t max_iters       = 20;
			float    mahal_thresh    = 4.0f;
			double   min_change_tran = 1.0e-3;
			double   min_change_rot  = 0.1*M_TAU/360.0;
		};

		struct GaussianMatchResults {
			Pose     pose;
			double   score;
		};

		inline PoseArray default_swarm() {
			PoseArray ret;
			ret.resize(1);
			ret(0) = Pose::Identity();
			return ret;
		}

	}

	struct GaussianModel {
		using FitParams    = detail::GaussianFitParams;
		using MatchParams  = detail::GaussianMatchParams;
		using MatchResults = detail::GaussianMatchResults;

		VecArray<3> centers;
		VecArray<3> log_scales;
		QuatArray   quats;

		Eigen::Index size() const {
			return centers.cols();
		}

		auto quats_array() {
			return Eigen::Map<VecArray<4>>{ quats.data()->coeffs().data(), Eigen::NoChange, quats.rows() };
		}

		auto quats_array() const {
			return Eigen::Map<VecArray<4> const>{ quats.data()->coeffs().data(), Eigen::NoChange, quats.rows() };
		}

		void transform(Pose const& pose) {
			for (Eigen::Index i = 0; i < size(); i ++) {
				centers.col(i) = pose*centers.col(i);
				quats(i) = pose.rotation()*quats(i);
				quats(i).normalize();
			}
		}

		void fit(AnyCloudIn cl, FitParams const& p);
		void fit_server(AnyCloudIn cl, FitParams const& p);

		bool match(
			MatchResults& out,
			AnyCloudIn cl,
			Pose const& init_pose,
			PoseArray const& init_particles,
			MatchParams const& p
		);

		template <typename PointType>
		void fit(pcl::PointCloud<PointType> const& cl, FitParams const& p = FitParams{}) {
			fit(cl.getMatrixXfMap(), p);
		}

		template <typename PointType>
		void fit_server(pcl::PointCloud<PointType> const& cl, FitParams const& p = FitParams{}) {
			fit_server(cl.getMatrixXfMap(), p);
		}

		template <typename PointType>
		bool match(
			MatchResults& out,
			pcl::PointCloud<PointType> const& cl,
			Pose const& init_pose = Pose::Identity(),
			PoseArray const& init_particles = detail::default_swarm(),
			MatchParams const& p = MatchParams{}
		) {
			return match(out, cl.getMatrixXfMap(), init_pose, init_particles, p);
		}
	};

}
