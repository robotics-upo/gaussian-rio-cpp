#include "gaussian_scan_matching.hpp"
#include <chrono>

namespace upo_gaussians {

namespace detail {

IcgContext::IcgContext(
	AnyCloudIn cl,
	GaussianModel const& model,
	Pose const& init_pose,
	PoseArray const& particles,
	Eigen::Index rcs_index
) :
	m_gm{model},
	m_initPose{init_pose},
	m_numPoints{(uint32_t)cl.cols()},
	m_numGaussians{(uint32_t)model.size()},
	m_numParticles{(uint32_t)particles.rows()},
	m_numBlocks{(m_numPoints + UPO_CUDA_NUM_THREADS - 1) / UPO_CUDA_NUM_THREADS}
{
	m_points.reserve(m_numPoints);
	for (uint32_t i = 0; i < m_numPoints; i ++) {
		m_points[i].segment<3>(0) = init_pose.cast<float>() * cl.col(i).segment<3>(0);
	}

	if (rcs_index >= 0 && model.has_rcs()) {
		m_pointRcs.resize(m_numPoints);
		m_pointRcs = cl.row(rcs_index).transpose();
	}

	m_g_centers.reserve(m_numGaussians);
	m_g_invscale.reserve(m_numGaussians);
	m_g_invrot.reserve(m_numGaussians);
	for (uint32_t i = 0; i < m_numGaussians; i ++) {
		m_g_centers[i].segment<3>(0) = model.centers.col(i).cast<float>();
		m_g_invscale[i].segment<3>(0) = (-model.log_scales.col(i).array()).exp().matrix().cast<float>();
		m_g_invrot[i] = model.quats(i).conjugate().cast<float>();
	}

	m_T_tran.reserve(m_numParticles);
	m_T_rot.reserve(m_numParticles);
	for (uint32_t i = 0; i < m_numParticles; i ++) {
		m_T_tran[i].segment<3>(0) = particles(i).translation().cast<float>();
		m_T_rot[i] = Quat(particles(i).rotation()).cast<float>();
	}

	m_matchups.reserve(m_numPoints*m_numParticles);
	m_sum_reduce_temp.reserve(m_numBlocks*m_numParticles);
}

std::pair<size_t,double> IcgContext::matchup(float max_mahal)
{
	cuda_matchupP2G(max_mahal);
	cudaDeviceSynchronize();

	size_t best_particle = 0;
	double best_sqmahal = 0.0;

	for (uint32_t i = 0; i < m_numParticles; i ++) {
		MatchOutd out = { 0.0, 0 };
		for (size_t j = 0; j < m_numBlocks; j ++) {
			auto& in = sr_matchOut()[j + m_numBlocks*i];
			out.sqmahal += sqrt(double(in.sqmahal));
			out.matches += in.matches;
		}

		if (i == 0 || out.sqmahal < best_sqmahal) {
			best_particle = i;
			best_sqmahal = out.sqmahal;
		}
	}

	return { best_particle, best_sqmahal / m_numPoints };
}

void IcgContext::iteration(double min_change_rot, double min_change_tran, double rcs_weight)
{
	for (size_t i = 0; i < m_numParticles; i ++) {
		if (is_converged(i)) {
			continue;
		}

		Mat<6> H = Mat<6>::Zero();
		Vec<6> b = Vec<6>::Zero();

		Mat<6> H_rcs = Mat<6>::Zero();
		Vec<6> b_rcs = Vec<6>::Zero();

		auto const* matchups = &m_matchups[i*m_numPoints];

		//printf("  # for particle %zu\n", i);

		Quat T_rot = m_T_rot[i].cast<double>();
		Vec<3> T_tran = m_T_tran[i].head<3>().cast<double>();

		for (size_t j = 0; j < m_numPoints; j ++) {
			int32_t gid = matchups[j];
			if (gid < 0) {
				continue;
			}

			Vec<3> transed_point = T_rot*m_points[j].head<3>().cast<double>() + T_tran;

			Vec<3> g_center = m_g_centers[gid].head<3>().cast<double>();
			Vec<3> g_invscale = m_g_invscale[gid].head<3>().cast<double>();
			Quat g_invrot = m_g_invrot[gid].cast<double>();

			Mat<3> M = g_invscale.asDiagonal() * g_invrot.toRotationMatrix();
			Mat<3> P = M.transpose() * M;

			Mat<3,6> J;
			J.block<3,3>(0,0).setIdentity();
			J.block<3,3>(0,3) = -skewsym(transed_point);

			H += J.transpose() * P * J;
			b += J.transpose() * P * (transed_point - g_center);

			if (m_pointRcs.size()) {
				float rcs_scale = m_gm.rcs_scales(gid);
				if (std::isnan(rcs_scale)) {
					continue;
				}

				Vec<3> ray_unnorm = transed_point - g_center;
				auto ray_sph = make_sph<GaussianModel::G_SPH_LEVEL,float>(ray_unnorm.normalized().cast<float>());
				double pred_rcs = m_gm.rcs_coefs.col(gid).dot(ray_sph);
				double real_rcs = std::pow(10.0, (m_pointRcs(j) - rcs_scale)/10.0);

				auto rcs_grad = make_rcs_gradient(transed_point, g_center, m_gm.rcs_coefs.col(gid).cast<double>());
				H_rcs += rcs_grad * rcs_grad.transpose();
				b_rcs += rcs_grad * (pred_rcs - real_rcs);
			}
		}

		//printf("  done, now calculating change\n");

		Vec<6> change = H.ldlt().solve(-b);
		//std::cout << "  change = [" << change.transpose() << "]" << std::endl;

		Vec<3> ch_tran = change.head<3>();
		Vec<3> ch_rot_so3 = change.tail<3>();
		Quat ch_rot = so3_exp(ch_rot_so3);

		m_T_rot[i] = (ch_rot*T_rot).cast<float>();
		m_T_tran[i].head<3>() = (ch_rot*T_tran + ch_tran).cast<float>();

		if (ch_tran.norm() <= min_change_tran && ch_rot_so3.norm() <= min_change_rot) {
			m_convergedParticles |= UINT64_C(1) << i;
		}
	}
}

}

std::vector<int32_t> GaussianModel::matchup(
	AnyCloudIn cl,
	float max_mahal
)
{
	PoseArray pa;
	pa.resize(1);
	pa(0) = Pose::Identity();
	detail::IcgContext icg{cl, *this, pa(0), pa};
	icg.matchup(max_mahal);

	std::vector<int32_t> ret((size_t)cl.cols());
	for (size_t i = 0; i < ret.size(); i ++) {
		ret[i] = icg.matchup_for(i);
	}

	return ret;
}

bool GaussianModel::match(
	MatchResults& out,
	AnyCloudIn cl,
	Pose const& init_pose,
	PoseArray const& init_particles,
	MatchParams const& p,
	Eigen::Index rcs_idx,
	RcsMatchParams const* rcsp
)
{
	if ((rcs_idx < 3) != !rcsp || rcs_idx >= cl.rows()) {
		return false;
	}

	detail::IcgContext icg{cl, *this, init_pose, init_particles, rcs_idx};
	size_t iter = 0;
	std::pair<size_t, double> best_so_far;

	icg.matchup(p.mahal_thresh);

	do {
		iter ++;
		icg.iteration(p.min_change_rot, p.min_change_tran, rcsp ? rcsp->rcs_weight : 0.0);
		best_so_far = icg.matchup(p.mahal_thresh);
	} while (iter < p.max_iters && icg.num_converged_particles() < icg.num_particles());

	auto [ best_particle, best_score ] = best_so_far;
	if (!icg.is_converged(best_particle)) {
		return false;
	}

	out.pose = icg.particle(best_particle);
	out.score = best_score;
	return true;
}

}
