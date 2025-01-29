#pragma once
#include "types.hpp"

namespace upo_gaussians {

	template <typename PointType, typename RandomEngine, typename Scalar = double>
	class BisectingKMeans {
		using CloudType = pcl::PointCloud<PointType>;
		using PosVector = Eigen::Vector<Scalar, 3>;
		using CenterMtx = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;

		struct Tree {
			std::vector<size_t> indices;
			Tree* left = nullptr;
			Tree* right = nullptr;
			PosVector center = PosVector::Zero();

			bool is_leaf() const { return !indices.empty(); }
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
				// Do nothing if we are too small
				if (indices.size() < 2) {
					return;
				}

				// Select two random points to serve as initial cluster centers
				size_t init_i = ctx.random_index(indices);
				size_t init_j = ctx.random_index(indices);
				while (init_j == init_i) {
					init_j = ctx.random_index(indices);
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

			void get_center(CenterMtx& out, Eigen::Index& i) const
			{
				if (is_leaf()) {
					out.col(i++) = center;
				} else {
					left->get_center(out, i);
					right->get_center(out, i);
				}
			}

		private:
			Scalar rebalance(BisectingKMeans& ctx)
			{
				Scalar cost = 0.0;
				PosVector lcenter = PosVector::Zero();
				PosVector rcenter = PosVector::Zero();

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

				left->center  = lcenter /  left->indices.size();
				right->center = rcenter / right->indices.size();

				return cost;
			}
		};

		CloudType const& m_cloud;
		RandomEngine& m_rng;
		std::vector<Tree> m_nodes;
		Scalar m_tol;

		PosVector point(size_t id) const {
			return m_cloud[id].getVector3fMap().template cast<Scalar>();
		}

		size_t random_index(std::vector<size_t> const& indices) {
			return indices[std::uniform_int_distribution<size_t>{0, indices.size()}(m_rng)];
		}

	public:
		BisectingKMeans(CloudType const& cl, RandomEngine& rng, size_t num_clusters, Scalar tol = 1.0e-4) :
			m_cloud{cl}, m_rng{rng}, m_tol{tol}
		{
			m_nodes.reserve(2*num_clusters - 1);

			auto& root = m_nodes.emplace_back();
			root.initialize(cl.size());

			while (--num_clusters) {
				root.candidate()->split(*this);
			}
		}

		void get_centers(CenterMtx& out) const
		{
			Eigen::Index i = 0;
			out.resize(Eigen::NoChange, (m_nodes.size()+1)/2);
			m_nodes[0].get_center(out, i);
		}

	};
}
