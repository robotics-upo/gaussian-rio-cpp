#pragma once
#include "types.hpp"

namespace upo_gaussians {

	class Traj {
		struct Point {
			double ts;
			Vec<3> pos;
			Quat quat;

			Pose pose() const { return make_pose(quat, pos); }
		};

		std::vector<Point> m_points;

	public:
		Traj() = default;
		Traj(Traj const&) = default;
		Traj(Traj&&) = default;

		bool empty() const { return m_points.empty(); }
		size_t size() const { return m_points.size(); }
		double start() const { return m_points.front().ts; }
		double end() const { return m_points.back().ts; }

		bool load(const char* fname);
		Pose operator()(double t) const;
		void recenter();

		template <typename T>
		void modify(T lambda) {
			for (auto& pt : m_points) {
				lambda(pt.pos, pt.quat);
			}
		}
	};

}
