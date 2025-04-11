#include "gaussian_scan_matching.hpp"
#include <chrono>

namespace upo_gaussians {

namespace detail {

IcgContext::IcgContext(
	AnyCloudIn cl,
	GaussianModel const& model,
	Pose const& init_pose,
	PoseArray const& particles
) :
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
			out.sqmahal += in.sqmahal;
			out.matches += in.matches;
		}

		if (i == 0 || out.sqmahal < best_sqmahal) {
			best_particle = i;
			best_sqmahal = out.sqmahal;
		}
	}

	return { best_particle, best_sqmahal / m_numPoints };
}

void IcgContext::iteration(double min_change_rot, double min_change_tran)
{
	for (size_t i = 0; i < m_numParticles; i ++) {
		if (m_convergedParticles & (UINT64_C(1) << i)) {
			continue;
		}

		Mat<6> H = Mat<6>::Zero();
		Vec<6> b = Vec<6>::Zero();

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

bool GaussianModel::match(
	MatchResults& out,
	AnyCloudIn cl,
	Pose const& init_pose,
	PoseArray const& init_particles,
	MatchParams const& p
)
{
	detail::IcgContext icg{cl, *this, init_pose, init_particles};
	{
		auto [ particle, score ] = icg.matchup(p.mahal_thresh);
		out.pose = icg.particle(particle);
		out.score = score;
	}

	size_t iter = 0;
	std::pair<size_t, double> best_so_far;
	do {
		iter ++;
		icg.iteration(p.min_change_rot, p.min_change_tran);
		best_so_far = icg.matchup(p.mahal_thresh);
	} while (iter < p.max_iters && icg.num_converged_particles() < icg.num_particles());

	auto [ best_particle, best_score ] = best_so_far;
	if (best_score >= out.score) {
		return false;
	}

	out.pose = icg.particle(best_particle);
	out.score = best_score;
	return true;
}

}
