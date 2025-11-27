#include <upo_gaussians/gaussian_model.hpp>
#include <upo_gaussians/sph.hpp>

namespace upo_gaussians {

void GaussianModel::fit_rcs(
	RadarCloud const& cl,
	RcsParams const& p
)
{
	size_t numg = size();
	auto gm_match = matchup(cl);

	std::vector<size_t> ppg(numg);
	std::vector<std::vector<Vec<16,float>>> g_shmtxs(numg);
	std::vector<std::vector<float>>         g_rcstgt(numg);
	rcs_scales.resize(numg);
	rcs_scales.setZero();

	for (size_t i = 0; i < gm_match.size(); i++) {
		int32_t gidx = gm_match[i];
		if (gidx < 0) continue;
		ppg[gidx]++;
		rcs_scales(gidx) += cl[i].power;
	}

	for (size_t i = 0; i < numg; i ++) {
		size_t sz = ppg[i];
		if (sz < p.min_ppg) continue;

		g_shmtxs[i].reserve(sz);
		g_rcstgt[i].reserve(sz);
		rcs_scales(i) /= sz;
	}

	for (size_t i = 0; i < gm_match.size(); i ++) {
		int32_t gidx = gm_match[i];
		if (gidx < 0 || ppg[gidx] < p.min_ppg) continue;

		float rcs = cl[i].power - rcs_scales(gidx);
		if (fabs(rcs) > p.db_thresh) continue;

		rcs = std::pow(10.0, rcs/10.0);

		Vec<3> invec = cl[i].getVector3fMap().cast<double>();
		if (p.use_incident) {
			invec = incident(gidx, invec);
		} else {
			invec -= centers.col(gidx);
		}

		g_shmtxs[gidx].emplace_back(make_sph<G_SPH_LEVEL,float>(invec.normalized().cast<float>()));
		g_rcstgt[gidx].emplace_back(rcs);
	}

	rcs_coefs.resize(Eigen::NoChange, numg);

	for (size_t i = 0; i < numg; i ++) {
		auto& shmtx = g_shmtxs[i];
		auto& rcstgt = g_rcstgt[i];

		if (rcstgt.empty()) continue;

		auto eig_shmtx = Eigen::Map<Eigen::MatrixXf>(shmtx.data()->data(), G_SPH_NCOEFS, shmtx.size()).transpose();
		auto eig_rcstgt = Eigen::Map<Eigen::VectorXf>(rcstgt.data(), rcstgt.size());

		auto solver = eig_shmtx.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
		if (solver.info() != Eigen::Success) {
			printf("SVD returned Pifia\n");
			continue;
		}

		rcs_coefs.col(i) = solver.solve(eig_rcstgt);
	}

}

}
