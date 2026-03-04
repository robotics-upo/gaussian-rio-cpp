#pragma once
#include "types.hpp"
#include "radar.hpp"
#include "sph.hpp"

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

		struct GaussianRcsParams {
			double   db_thresh      = 10.0;
			uint16_t min_ppg        = 16;
			bool     use_incident   = false;
		};

		struct GaussianMatchParams {
			uint32_t max_iters       = 20;
			float    mahal_thresh    = 4.0f;
			double   min_change_tran = 1.0e-3;
			double   min_change_rot  = 0.1*M_TAU/360.0;
		};

		struct GaussianRcsMatchParams {
			double   rcs_weight      = 0.01;
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
		using RcsParams    = detail::GaussianRcsParams;
		using MatchParams  = detail::GaussianMatchParams;
		using RcsMatchParams = detail::GaussianRcsMatchParams;
		using MatchResults = detail::GaussianMatchResults;

		static constexpr unsigned G_SPH_LEVEL  = 3;
		static constexpr unsigned G_SPH_NCOEFS = SPH_NUM_COEFS<G_SPH_LEVEL>;

		VecArray<3> centers;
		VecArray<3> log_scales;
		QuatArray   quats;

		DynVecf     rcs_scales;
		VecArray<G_SPH_NCOEFS,float> rcs_coefs;

		Eigen::Index size() const {
			return centers.cols();
		}

		bool has_rcs() const {
			return rcs_scales.size() != 0;
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

		Vec<3> incident(size_t gidx, Vec<3> const& v) {
			return (-log_scales.col(gidx)).array().exp()*(quats(gidx).conjugate()*(v - centers.col(gidx))).array();
		}

		void fit(AnyCloudIn cl, FitParams const& p);
		void fit_server(AnyCloudIn cl, FitParams const& p);
		void fit_ablation(AnyCloudIn cl);

		std::vector<int32_t> matchup(AnyCloudIn cl, float max_mahal);

		void fit_rcs(RadarCloud const& cl, IncidenceCloud const& incicl, RcsParams const& p = RcsParams{});

		void save(const char* fname) const;
		bool load(const char* fname);

		bool match(
			MatchResults& out,
			AnyCloudIn cl,
			Pose const& init_pose,
			PoseArray const& init_particles,
			MatchParams const& p,
			Eigen::Index rcs_idx = -1,
			RcsMatchParams const* rcsp = nullptr
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
		void fit_ablation(pcl::PointCloud<PointType> const& cl) {
			fit_ablation(cl.getMatrixXfMap());
		}

		template <typename PointType>
		std::vector<int32_t> matchup(pcl::PointCloud<PointType> const& cl, float max_mahal = 100.0f) {
			return matchup(cl.getMatrixXfMap(), max_mahal);
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

		bool match_rcs(
			MatchResults& out,
			RadarCloud const& cl,
			Pose const& init_pose = Pose::Identity(),
			PoseArray const& init_particles = detail::default_swarm(),
			MatchParams const& p = MatchParams{},
			RcsMatchParams const& rcsp = RcsMatchParams{}
		) {
			return match(out, cl.getMatrixXfMap(), init_pose, init_particles, p,
				offsetof(RadarPoint,power)/sizeof(float), &rcsp);
		}
	};

}
