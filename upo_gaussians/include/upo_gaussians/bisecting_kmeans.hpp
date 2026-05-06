#pragma once
#include "types.hpp"

namespace upo_gaussians {

	template <typename RandomEngine, typename Scalar = double>
	class BisectingKMeans {
		using Vector    = Eigen::Vector<Scalar, 3>;
		using Quat      = Eigen::Quaternion<Scalar>;
		using VectorMtx = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;
		using QuatMtx   = Eigen::Vector<Quat, Eigen::Dynamic>;

		struct Tree {
			std::vector<size_t> indices;
			Tree* left = nullptr;
			Tree* right = nullptr;
			Vector center = Vector::Zero();

			bool is_leaf() const { return !left || !right; }
			size_t size() const { return indices.size(); }

			void initialize(size_t num_points)
			{
				indices.reserve(num_points);
				for (size_t i = 0; i < num_points; i ++) {
					indices.push_back(i);
				}
			}

			Tree* candidate()
			{
				if (is_leaf()) {
					return this;
				} else {
					Tree* a = left->candidate();
					Tree* b = right->candidate();
					return a->size() >= b->size() ? a : b;
				}
			}

			void split(BisectingKMeans& ctx)
			{
				// Select two random points to serve as initial cluster centers
				size_t init_i = ctx.random_index(indices);
				size_t init_j = ctx.random_index(indices);
				size_t attempts = 0;
				while (attempts < 10 && ctx.points_are_too_close(init_i, init_j)) {
					init_j = ctx.random_index(indices);
					attempts ++;
				}

				if (attempts == 10) {
					fprintf(stderr, "BisectingKMeans::split(): cluster resisted splitting\n");
					for (size_t idx : indices) {
						std::cerr << "[" << idx << "]: " << ctx.point(idx).transpose() << std::endl;
					}
					std::abort();
				}

				// Create and initialize leaves
				size_t heuristic_sz = (indices.size()+1)/2;
				left  = &ctx.m_nodes.emplace_back();
				right = &ctx.m_nodes.emplace_back();
				left->indices .reserve(heuristic_sz);
				right->indices.reserve(heuristic_sz);
				left->center  = ctx.point(init_i);
				right->center = ctx.point(init_j);

				// Optimize leaf centers
				Scalar cur_cost = rebalance(ctx);
				Scalar new_cost;
				while (cur_cost - (new_cost = rebalance(ctx)) >= ctx.m_tol) {
					cur_cost = new_cost;
				}

				// Mark ourselves as a non-leaf
				indices.clear();
				indices.shrink_to_fit();
			}

			void get_center(VectorMtx& out, Eigen::Index& i) const
			{
				if (is_leaf()) {
					out.col(i++) = center;
				} else {
					left->get_center(out, i);
					right->get_center(out, i);
				}
			}

			void get_scalerot(BisectingKMeans const& ctx, VectorMtx& out_scales, QuatMtx& out_quats, Eigen::Index& i, Scalar min_logsize) const
			{
				if (is_leaf()) {
					calc_scalerot(ctx, out_scales.col(i).data(), out_quats(i), min_logsize);
					++i;
				} else {
					left->get_scalerot(ctx, out_scales, out_quats, i, min_logsize);
					right->get_scalerot(ctx, out_scales, out_quats, i, min_logsize);
				}
			}

		private:
			Scalar rebalance(BisectingKMeans& ctx)
			{
				Scalar cost = 0.0;
				Vector lcenter = Vector::Zero();
				Vector rcenter = Vector::Zero();

				left->indices.clear();
				right->indices.clear();

				for (size_t id : indices) {
					auto pt = ctx.point(id);

					Scalar lcost = (pt -  left->center).squaredNorm();
					Scalar rcost = (pt - right->center).squaredNorm();

					if (lcost <= rcost) {
						cost += lcost;
						lcenter += pt;
						left->indices.push_back(id);
					} else {
						cost += rcost;
						rcenter += pt;
						right->indices.push_back(id);
					}
				}

				if (!left->indices.size()) {
					fprintf(stderr, "BisectingKMeans::rebalance(): left cluster emptied out\n");
					std::cerr << "L: [" << left->center.transpose() << "]" << std::endl;
					std::cerr << "R: [" << right->center.transpose() << "]" << std::endl;
					std::abort();
				}

				if (!right->indices.size()) {
					fprintf(stderr, "BisectingKMeans::rebalance(): right cluster emptied out\n");
					std::cerr << "L: [" << left->center.transpose() << "]" << std::endl;
					std::cerr << "R: [" << right->center.transpose() << "]" << std::endl;
					std::abort();
				}

				left->center  = lcenter /  left->indices.size();
				right->center = rcenter / right->indices.size();

				return cost;
			}

			void calc_scalerot(BisectingKMeans const& ctx, Scalar* logscale, Quat& rot, Scalar min_logsize) const
			{
				Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pts;
				pts.resize(Eigen::NoChange, indices.size());
				for (size_t i = 0; i < indices.size(); i ++) {
					pts.col(i) = ctx.point(indices[i]) - center;
				}

				auto svd = (pts*pts.transpose()).jacobiSvd(Eigen::ComputeFullU);
				Eigen::Map<Vector>{logscale} = (Scalar{0.5}*svd.singularValues().array().log()).max(min_logsize).matrix();
				rot = svd.matrixU().transpose() / svd.matrixU().determinant();
			}
		};

		AnyCloudIn m_cloud;
		RandomEngine& m_rng;
		std::vector<Tree> m_nodes;
		Scalar m_tol;

		Vector point(size_t id) const {
			return m_cloud.col(id).segment<3>(0).cast<Scalar>();
		}

		size_t random_index(std::vector<size_t> const& indices) {
			return indices[std::uniform_int_distribution<size_t>{0, indices.size()-1}(m_rng)];
		}

		bool points_are_too_close(size_t i, size_t j) {
			if (i == j) {
				return true;
			}

			return (point(j)-point(i)).squaredNorm() < 1e-6;
		}

	public:
		template <typename PointType>
		BisectingKMeans(pcl::PointCloud<PointType> const& cl, RandomEngine& rng, size_t max_clusters, size_t min_points = 2, Scalar tol = Scalar{1.0e-4}) :
			BisectingKMeans{cl.getMatrixXfMap(), rng, max_clusters, min_points, tol} { }

		BisectingKMeans(AnyCloudIn cl, RandomEngine& rng, size_t max_clusters, size_t min_points = 2, Scalar tol = Scalar{1.0e-4}) :
			m_cloud{cl}, m_rng{rng}, m_tol{tol}
		{
			m_nodes.reserve(2*max_clusters - 1);

			auto& root = m_nodes.emplace_back();
			root.initialize(cl.cols());

			while (--max_clusters) {
				auto leaf = root.candidate();
				if (leaf->size() < 2*min_points) {
					break;
				}
				leaf->split(*this);
			}
		}

		void get_centers(VectorMtx& out) const
		{
			Eigen::Index i = 0;
			out.resize(Eigen::NoChange, (m_nodes.size()+1)/2);
			m_nodes[0].get_center(out, i);
		}

		void get_scalerot(VectorMtx& out_scales, QuatMtx& out_quats, Scalar min_logsize = Scalar{-3.0}) const
		{
			Eigen::Index i = 0;
			out_scales.resize(Eigen::NoChange, (m_nodes.size()+1)/2);
			out_quats.resize((m_nodes.size()+1)/2);
			m_nodes[0].get_scalerot(*this, out_scales, out_quats, i, min_logsize);
		}

	};
}
