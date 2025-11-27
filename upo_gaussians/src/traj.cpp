#include <upo_gaussians/traj.hpp>
#include <stdio.h>

namespace upo_gaussians {

bool Traj::load(const char* fname)
{
	FILE* f = fopen(fname, "r");
	if (!f) {
		return false;
	}

	char* linebuf = nullptr;
	size_t linebufsz = 0;
	ssize_t linelen;
	while ((linelen = getline(&linebuf, &linebufsz, f)) >= 0) {
		Point pt;
		int suc = sscanf(linebuf, "%lf %lf %lf %lf %lf %lf %lf %lf",
			&pt.ts,
			&pt.pos(0), &pt.pos(1), &pt.pos(2),
			&pt.quat.x(), &pt.quat.y(), &pt.quat.z(), &pt.quat.w()
		);
		if (suc == 8 && (empty() || end() <= pt.ts)) {
			m_points.emplace_back(std::move(pt));
		}
	}

	fclose(f);
	return true;
}

Pose Traj::operator()(double t) const
{
	if (empty()) {
		return Pose::Identity();
	}

	if (t <= start()) {
		return m_points.front().pose();
	}

	if (t >= end()) {
		return m_points.back().pose();
	}

	// Binary search
	size_t L = 0, R = size();
	while (L < R) {
		size_t M = (L+R)/2;
		if (m_points[M].ts < t) {
			L = M+1;
		} else {
			R = M;
		}
	}

	// Interpolation
	auto& pt_a = m_points[L-1];
	auto& pt_b = m_points[L];
	double interp   = (t - pt_a.ts) / (pt_b.ts - pt_a.ts);
	Vec<3> int_pos  = (1.0-interp)*pt_a.pos + interp*pt_b.pos;
	Quat   int_quat = pt_a.quat.slerp(interp, pt_b.quat);
	return make_pose(int_quat, int_pos);
}

void Traj::recenter()
{
	if (empty()) return;

	Vec<3> ref_p = m_points[0].pos;
	Quat   ref_q = m_points[0].quat;

	//Quat deyaw = Quat{ref_q.w(), ref_q.x(), ref_q.y(), 0.0}.normalized().conjugate();
	double yaw = atan2(
		2*(ref_q.w()*ref_q.z() + ref_q.x()*ref_q.y()),
		1 - 2*(ref_q.y()*ref_q.y() + ref_q.z()*ref_q.z())
	);

	Quat deyaw = Quat{Eigen::AngleAxisd{-yaw, Vec<3>{0.0,0.0,1.0}}};

	modify([&](Vec<3>& p, Quat& q) {
		p = deyaw*(p-ref_p);
		q = deyaw*q;
	});
}

}
