#include "gaussian_scan_matching.hpp"

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
	//printf("m_numParticles = %u\n", m_numParticles);
	//printf("m_numPoints    = %u\n", m_numPoints);
	//printf("m_numGaussians = %u\n", m_numGaussians);
	//printf("m_numBlocks    = %u\n", m_numBlocks);

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
	m_xy0.reserve(m_numParticles);
	m_sum_reduce_temp.reserve(m_numBlocks*m_numParticles);
	m_sum_reduce_temp2.reserve(m_numBlocks*m_numParticles);
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

void IcgContext::iteration()
{
	// Overall workflow:
	// Compute x0,y0 centroids:
	//   (Σ P_i) [x0 y0] = Σ P_i [x_i y_i]
	// Compute rotation:
	//   u_i = x_i - x0; v_i = y_i - y0
	//   Σ P_i R u_i u_i^T = Σ P_i v_i u_i^T
	//   Solve iteratively using Newton's method and SO(3) -> so(3) tangential space:
	//     R_{k+1} = (I + skewsym(δθ)) * R_k
	//     F = Σ P_i skewsym(δθ) R_k u_i u_i^T + Σ P_i (R_k u_i - v_i) u_i^T = 0
	//       ∂F/∂δθ * δθ  = -F
	//       9x3      3x1 = 9x1

	cuda_sumReducePxy();
	cudaDeviceSynchronize();
	for (size_t i = 0; i < m_numParticles; i ++) {
		SPMat3<double> symP;
		MultiVec<3,2> xy = MultiVec<3,2>::Zero();
		for (size_t j = 0; j < m_numBlocks; j ++) {
			symP += sr_pmat()[j + m_numBlocks*i].cast<double>();
			xy   += sr_xy0() [j + m_numBlocks*i].block(0,0,3,2).cast<double>();
		}

		Mat<3> P = symP;
		MultiVec<3,2> xy0 = P.ldlt().solve(xy);
		m_xy0[i].block(0,0,3,2) = xy0.cast<float>();

		/*
		printf("Sum P =[ %f, %f, %f\n",   P(0,0), P(0,1), P(0,2));
		printf("         %f, %f, %f\n",   P(1,0), P(1,1), P(1,2));
		printf("         %f, %f, %f ]\n", P(2,0), P(2,1), P(2,2));
		printf("Sum x = [%f, %f, %f]\n", xy(0,0), xy(1,0), xy(2,0));
		printf("Sum y = [%f, %f, %f]\n", xy(0,1), xy(1,1), xy(2,1));
		printf("   x0 = [%f, %f, %f]\n", xy0(0,0), xy0(1,0), xy0(2,0));
		printf("   y0 = [%f, %f, %f]\n", xy0(0,1), xy0(1,1), xy0(2,1));
		*/
	}

	for (unsigned iter = 0; iter < 4; iter ++) {
		//printf(" -- iter %u\n", iter);
		cuda_sumReduceRotOpt();
		cudaDeviceSynchronize();

		for (size_t i = 0; i < m_numParticles; i ++) {
			Mat<12,3> M = Mat<12,3>::Zero();
			for (size_t j = 0; j < m_numBlocks; j ++) {
				M += sr_rotopt()[j + m_numBlocks*i].cast<double>();
			}

			auto F = M.block(0,0,3,3).reshaped();
			auto J = M.block(3,0,9,3);

			Vec<3> dtheta = J.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(F);
			Quat q = so3_exp(dtheta) * m_T_rot[i].cast<double>();
			q.normalize();
			m_T_rot[i] = q.cast<float>();

			//printf(" dz = [%f, %f, %f]\n", dtheta(0), dtheta(1), dtheta(2));
			//printf("  q = [%f, %f, %f, %f]\n", q.w(), q.x(), q.y(), q.z());
		}
	}

	for (size_t i = 0; i < m_numParticles; i ++) {
		auto    xy0 = m_xy0[i].block(0,0,3,2).cast<double>();
		auto      q = m_T_rot[i].cast<double>();
		Vec<3> tran = xy0.col(1) - q*xy0.col(0);
		m_T_tran[i].segment<3>(0) = tran.cast<float>();

		//printf("  t = [%f, %f, %f]\n",     tran(0), tran(1), tran(2));
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
		//printf("!! Initial score: %f\n", out.score);
	}

	size_t impatience = 0;
	bool did_anything = false;
	do {
		icg.iteration();
		auto [particle, score] = icg.matchup(p.mahal_thresh);

		if (score + p.min_improvement < out.score) {
			//printf("!! Improved to: %f\n", score);
			out.pose = icg.particle(particle);
			out.score = score;
			impatience = 0;
			did_anything = true;
		} else {
			//printf("!! Worsened to: %f\n", score);
			impatience ++;
		}

	} while (impatience < p.patience);

	return did_anything;
}

}
