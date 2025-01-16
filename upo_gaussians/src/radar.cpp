#include <upo_gaussians/radar.hpp>

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

#include <iostream>

namespace upo_gaussians {

namespace {

	bool median_check(
		EgoVelResult& ret,
		RadarCloud const& in_cloud,
		EgoVelParams const& p
	)
	{
		std::vector<float> mdop;
		mdop.reserve(in_cloud.size());
		for (auto& pt : in_cloud) {
			mdop.push_back(fabsf(pt.doppler));
		}

		// Median check
		size_t mid = mdop.size()/2;
		std::nth_element(mdop.begin(), mdop.begin()+mid, mdop.end());
		if (mdop[mid] >= p.still_thresh) {
			return false;
		}

		ret.egovel.setZero();
		ret.egovel_cov.setZero();
		ret.egovel_cov.diagonal().fill((double)p.still_std*p.still_std);

		for (auto& pt : in_cloud) {
			if (fabsf(pt.doppler) < p.still_thresh) {
				ret.inliers.push_back(pt);
			}
		}

		return true;
	}

	bool solve_egovel_lsq(
		Vecf<3>& out_vel,
		DynMatf const& dirdops
	)
	{
		Eigen::Index pop_size = dirdops.rows();
		auto dirs = dirdops.block(0, 0, pop_size, 3);
		auto dops = dirdops.block(0, 3, pop_size, 1);

		// See https://eigen.tuxfamily.org/dox/group__LeastSquares.html
#ifdef WANT_LSQ_SOLVED_WITH_SVD
		auto solver = dirs.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
#else
		auto solver = (dirs.transpose()*dirs).ldlt();
#endif
		if (solver.info() != Eigen::Success) {
			return false;
		}

#ifdef WANT_LSQ_SOLVED_WITH_SVD
		out_vel = solver.solve(dops);
#else
		out_vel = solver.solve(dirs.transpose()*dops);
#endif
		return true;
	}

	size_t evaluate_egovel(
		DynMatf const& dirdops,
		Vecf<3> const& vel,
		DynVecf& out_error,
		float inlier_thresh
	)
	{
		size_t num_inliers = 0;
		size_t pop_size = dirdops.rows();
		auto dirs = dirdops.block(0, 0, pop_size, 3);
		auto dops = dirdops.block(0, 3, pop_size, 1);

		out_error = (dirs*vel - dops).array().abs();
		for (size_t i = 0; i < pop_size; i ++) {
			if (out_error(i) < inlier_thresh) {
				num_inliers ++;
			}
		}

		return num_inliers;
	}

	size_t solve_egovel_ransac(
		Vecf<3>& best_vel,
		DynVecf& best_err,
		DynMatf const& dirdops,
		EgoVelParams const& p
	)
	{
		size_t pop_size = dirdops.rows();
		size_t num_points = p.num_points <= pop_size ? p.num_points : pop_size;

		// XX: Mersenne Twister sucks! TODO: implement a better rng
		//std::mt19937 g{std::random_device{}()};
		std::mt19937 g{3135134162};

		std::vector<size_t> shuffle_buf( pop_size );
		std::iota(shuffle_buf.begin(), shuffle_buf.end(), 0);

		DynMatf lsq_buf{ num_points, dirdops.cols() };
		DynVecf err_buf;
		size_t best_inliers = 0;

		// etc etc
		unsigned num_iters = p.calc_num_iters();
		for (unsigned i = 0; i < num_iters; i ++) {
			std::shuffle(shuffle_buf.begin(), shuffle_buf.end(), g);
			for (unsigned j = 0; j < num_points; j ++) {
				lsq_buf.row(j) = dirdops.row(shuffle_buf[j]);
			}

			Vecf<3> cur_vel;
			if (!solve_egovel_lsq(cur_vel, lsq_buf)) {
				continue;
			}

			size_t cur_inliers = evaluate_egovel(dirdops, cur_vel, err_buf, p.inlier_thresh);
			if (cur_inliers > best_inliers) {
				best_vel = cur_vel;
				best_err = err_buf;
				best_inliers = cur_inliers;
			}

			if (cur_inliers == pop_size) {
				// Already saturated
				break;
			}
		}

		if (!best_inliers || (float)best_inliers/pop_size < p.min_inlier_p) {
			return 0;
		}

		return best_inliers;
	}

}

RadarCloud filter_radar_cloud(
	RadarCloud const& in_cloud,
	RadarFilterParams const& p
)
{
	RadarCloud out;
	out.reserve(in_cloud.size());

	for (auto& pt : in_cloud) {
		float azimuth = atan2f(pt.y, pt.x);
		float elevation = atan2f(pt.z, sqrtf(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z));
		if (pt.power >= p.min_power && fabsf(azimuth) < p.max_azimuth && fabsf(elevation) < p.max_elevation) {
			out.push_back(pt);
		}
	}

	return out;
}

bool calc_radar_egovel(
	EgoVelResult& ret,
	RadarCloud const& in_cloud,
	EgoVelParams const& p
)
{
	if (median_check(ret, in_cloud, p)) {
		return true;
	}

	DynMatf dirdops{ in_cloud.size(), 4 };
	for (size_t i = 0; i < in_cloud.size(); i ++) {
		auto& p = in_cloud[i];
		dirdops.block(i, 0, 1, 3) = p.getVector3fMap().normalized().transpose();
		dirdops(i, 3) = p.doppler;
	}

	Vecf<3> best_vel;
	DynVecf best_err;
	size_t num_inliers = solve_egovel_ransac(best_vel, best_err, dirdops, p);
	if (!num_inliers) {
		return false;
	}

	DynMatf inlier_dirdops;
	inlier_dirdops.resize(num_inliers, 4);
	ret.inliers.reserve(num_inliers);
	for (size_t i = 0; i < in_cloud.size(); i ++) {
		if (best_err(i) < p.inlier_thresh) {
			inlier_dirdops.row(ret.inliers.size()) = dirdops.row(i);
			ret.inliers.push_back(in_cloud[i]);
		}
	}

	if (ret.inliers.size() != num_inliers) {
		// Shouldn't happen
		std::cerr << "  !! DO NOT MATCH !! " << ret.inliers.size() << " vs " << num_inliers << std::endl;
		std::abort();
	}

	if (!solve_egovel_lsq(best_vel, inlier_dirdops)) {
		std::cerr << "  !! egovel lsq recalc fail" << std::endl;
		return false;
	}

	auto inlier_dirs = inlier_dirdops.block(0, 0, num_inliers, 3);
	Matf<3> HtH = inlier_dirs.transpose()*inlier_dirs;
	evaluate_egovel(inlier_dirdops, best_vel, best_err, p.inlier_thresh);

	ret.egovel = -best_vel.cast<double>(); // flip sign in order to refer to ourselves moving
	ret.egovel_cov = HtH.cast<double>().inverse();
	ret.egovel_cov *= best_err.dot(best_err) / (num_inliers + 3);

	return true;
}

}
