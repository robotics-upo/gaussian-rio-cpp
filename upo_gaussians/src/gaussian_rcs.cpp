#include <upo_gaussians/gaussian_model.hpp>
#include <upo_gaussians/sph.hpp>

namespace upo_gaussians {

namespace {

void matlab_rcsSphGradient(
	const double in1[3], const double in2[3], const double in3[16],
	const double in4[3], const double in5[4], double vnorm,
	double fun[6])
{
	double b_fun_tmp;
	double b_t135_tmp;
	double c_fun_tmp;
	double d_fun_tmp;
	double e_fun_tmp;
	double f_fun_tmp;
	double fun_tmp;
	double g_fun_tmp;
	double h_fun_tmp;
	double i_fun_tmp;
	double j_fun_tmp;
	double k_fun_tmp;
	double l_fun_tmp;
	double m_fun_tmp;
	double n_fun_tmp;
	double o_fun_tmp;
	double p_fun_tmp;
	double q_fun_tmp;
	double r_fun_tmp;
	double t10;
	double t102;
	double t103;
	double t104;
	double t107;
	double t108;
	double t110;
	double t117;
	double t118;
	double t119;
	double t120;
	double t121;
	double t123;
	double t126;
	double t126_tmp;
	double t128;
	double t128_tmp;
	double t131;
	double t131_tmp;
	double t132;
	double t132_tmp;
	double t135;
	double t135_tmp;
	double t136;
	double t136_tmp;
	double t137;
	double t137_tmp;
	double t138;
	double t139;
	double t14;
	double t140;
	double t140_tmp;
	double t141;
	double t141_tmp;
	double t146;
	double t148;
	double t15;
	double t16;
	double t17;
	double t18;
	double t19;
	double t23;
	double t24;
	double t25;
	double t26;
	double t27;
	double t28;
	double t29;
	double t30;
	double t31;
	double t45;
	double t46;
	double t47;
	double t5;
	double t6;
	double t7;
	double t8;
	double t9;
	t5 = in5[0] * in5[1] * 2.0;
	t6 = in5[0] * in5[2] * 2.0;
	t7 = in5[0] * in5[3] * 2.0;
	t8 = in5[1] * in5[2] * 2.0;
	t9 = in5[1] * in5[3] * 2.0;
	t10 = in5[2] * in5[3] * 2.0;
	t14 = 1.0 / vnorm;
	t15 = t14 * t14;
	t16 = std::pow(t14, 3.0);
	t17 = in5[0] * in5[0] * 2.0;
	t18 = in5[1] * in5[1] * 2.0;
	t19 = in5[2] * in5[2] * 2.0;
	t23 = in4[0] - in2[0];
	t24 = in4[1] - in2[1];
	t25 = in4[2] - in2[2];
	t26 = t5 + t10;
	t27 = t6 + t9;
	t28 = t7 + t8;
	t29 = t5 - t10;
	t30 = t6 - t9;
	t31 = t7 - t8;
	t45 = (t17 + t18) - 1.0;
	t46 = (t17 + t19) - 1.0;
	t47 = (t18 + t19) - 1.0;
	t102 = ((in1[0] + t24 * t26) + t25 * t30) - t23 * t47;
	t103 = ((in1[1] + t25 * t28) + t23 * t29) - t24 * t46;
	t104 = ((in1[2] + t23 * t27) - t24 * t31) - t25 * t45;
	t107 = ((in4[1] * t27 + in4[0] * t31) - t24 * t27) - t23 * t31;
	t108 = ((in4[0] * t28 - in4[2] * t29) + t25 * t29) - t23 * t28;
	t118 = ((in4[2] * t27 + in4[0] * t45) - t25 * t27) - t23 * t45;
	t119 = ((in4[0] * t26 + in4[1] * t47) - t23 * t26) - t24 * t47;
	t120 = ((in4[1] * t28 + in4[2] * t46) - t24 * t28) - t25 * t46;
	t121 = ((in4[1] * t29 + in4[0] * t46) - t24 * t29) - t23 * t46;
	t123 = ((in4[0] * t30 + in4[2] * t47) - t23 * t30) - t25 * t47;
	t110 = t104 * t104;
	t7 = t15 * t26 * t102;
	t8 = t7 * 2.0;
	t126_tmp = t15 * t28 * t103;
	t126 = t126_tmp * 2.0;
	t17 = t15 * t29 * t103;
	t18 = t17 * 2.0;
	t128_tmp = t15 * t30 * t102;
	t128 = t128_tmp * 2.0;
	t19 = t15 * t46 * t103;
	t23 = t19 * 2.0;
	t131_tmp = t15 * t47 * t102;
	t131 = t131_tmp * 2.0;
	t5 = t15 * t103;
	t132_tmp = t5 * t108;
	t132 = t132_tmp * 2.0;
	t10 = t15 * t102;
	t135_tmp = ((in4[2] * t26 - in4[1] * t30) - t25 * t26) + t24 * t30;
	b_t135_tmp = t10 * t135_tmp;
	t135 = b_t135_tmp * 2.0;
	t136_tmp = t10 * t119;
	t136 = t136_tmp * 2.0;
	t137_tmp = t5 * t120;
	t137 = t137_tmp * 2.0;
	t140_tmp = t5 * t121;
	t140 = t140_tmp * 2.0;
	t141_tmp = t10 * t123;
	t141 = t141_tmp * 2.0;
	t5 = t15 * (t102 * t102);
	t10 = t15 * (t103 * t103);
	t117 = t15 * t110 * 5.0;
	t6 = t8 + t23;
	t9 = t18 + t131;
	t146 = t135 + t137;
	t148 = t136 + t140;
	t138 = t5 - t10;
	t139 = t5 - t10 * 3.0;
	fun_tmp = in3[5] * t15;
	b_fun_tmp = in3[4] * t15;
	c_fun_tmp = in3[7] * t15;
	d_fun_tmp = in3[14] * t14;
	e_fun_tmp = in3[15] * t14;
	f_fun_tmp = in3[9] * t14;
	g_fun_tmp = in3[10] * t16;
	h_fun_tmp = in3[3] * t14;
	i_fun_tmp = in3[2] * t14;
	j_fun_tmp = in3[1] * t14;
	k_fun_tmp = in3[6] * t15;
	l_fun_tmp = in3[13] * t14;
	m_fun_tmp = in3[11] * t14;
	n_fun_tmp = d_fun_tmp * t104;
	o_fun_tmp = f_fun_tmp * t103;
	p_fun_tmp = e_fun_tmp * t102;
	q_fun_tmp = t10 - t5 * 3.0;
	t10 = in3[11] * t16;
	r_fun_tmp = in3[13] * t16;
	fun[0] = (((((((((((((((((((((((-in3[12] * (t14 * t27 * 3.0 -
												t16 * t27 * t110 * 15.0) -
									in3[8] * t9) +
									i_fun_tmp * t27) +
									j_fun_tmp * t29) -
								h_fun_tmp * t47) +
								b_fun_tmp * t29 * t102) +
								fun_tmp * t27 * t103) +
								c_fun_tmp * t27 * t102) +
							k_fun_tmp * t27 * t104 * 6.0) +
							fun_tmp * t29 * t104) -
							b_fun_tmp * t47 * t103) -
							c_fun_tmp * t47 * t104) +
						m_fun_tmp * t29 * (t117 - 1.0)) +
						d_fun_tmp * t27 * t138) -
						l_fun_tmp * t47 * (t117 - 1.0)) -
						e_fun_tmp * t47 * t139) -
					n_fun_tmp * t9) -
					p_fun_tmp * (t131 + t17 * 6.0)) -
					o_fun_tmp * (t18 + t131_tmp * 6.0)) -
					f_fun_tmp * t29 * q_fun_tmp) +
				g_fun_tmp * t27 * t102 * t103) +
				g_fun_tmp * t29 * t102 * t104) +
				t10 * t27 * t103 * t104 * 10.0) +
				r_fun_tmp * t27 * t102 * t104 * 10.0) -
			g_fun_tmp * t47 * t103 * t104;
	fun[1] = (((((((((((((((((((((((in3[12] * (t14 * t31 * 3.0 -
												t16 * t31 * t110 * 15.0) +
									in3[8] * t6) +
									h_fun_tmp * t26) -
									i_fun_tmp * t31) -
								j_fun_tmp * t46) +
								b_fun_tmp * t26 * t103) +
								c_fun_tmp * t26 * t104) -
								fun_tmp * t31 * t103) -
							c_fun_tmp * t31 * t102) -
							k_fun_tmp * t31 * t104 * 6.0) -
							b_fun_tmp * t46 * t102) -
							fun_tmp * t46 * t104) +
						l_fun_tmp * t26 * (t117 - 1.0)) +
						e_fun_tmp * t26 * t139) -
						m_fun_tmp * t46 * (t117 - 1.0)) -
						d_fun_tmp * t31 * t138) +
					n_fun_tmp * t6) +
					o_fun_tmp * (t23 + t7 * 6.0)) +
					p_fun_tmp * (t8 + t19 * 6.0)) +
					f_fun_tmp * t46 * q_fun_tmp) +
				g_fun_tmp * t26 * t103 * t104) -
				g_fun_tmp * t31 * t102 * t103) -
				t10 * t31 * t103 * t104 * 10.0) -
				r_fun_tmp * t31 * t102 * t104 * 10.0) -
			g_fun_tmp * t46 * t102 * t104;
	t5 = t126 - t128;
	fun[2] = (((((((((((((((((((((((in3[12] * (t14 * t45 * 3.0 -
												t16 * t45 * t110 * 15.0) -
									in3[8] * t5) +
									j_fun_tmp * t28) +
									h_fun_tmp * t30) -
								i_fun_tmp * t45) +
								b_fun_tmp * t28 * t102) +
								b_fun_tmp * t30 * t103) +
								fun_tmp * t28 * t104) +
							c_fun_tmp * t30 * t104) -
							fun_tmp * t45 * t103) -
							c_fun_tmp * t45 * t102) -
							k_fun_tmp * t45 * t104 * 6.0) +
						m_fun_tmp * t28 * (t117 - 1.0)) +
						l_fun_tmp * t30 * (t117 - 1.0)) +
						e_fun_tmp * t30 * t139) -
						d_fun_tmp * t45 * t138) -
					o_fun_tmp * (t126 - t128_tmp * 6.0)) +
					p_fun_tmp * (t128 - t126_tmp * 6.0)) -
					f_fun_tmp * t28 * q_fun_tmp) -
					n_fun_tmp * t5) +
				g_fun_tmp * t28 * t102 * t104) +
				g_fun_tmp * t30 * t103 * t104) -
				g_fun_tmp * t45 * t102 * t103) -
				t10 * t45 * t103 * t104 * 10.0) -
			r_fun_tmp * t45 * t102 * t104 * 10.0;
	t23 = ((in4[2] * t31 - in4[1] * t45) - t25 * t31) + t24 * t45;
	t5 = g_fun_tmp * t102;
	t15 = t16 * t110;
	t47 = b_fun_tmp * t102;
	t126_tmp = fun_tmp * t104;
	t131_tmp = fun_tmp * t103;
	t131 = c_fun_tmp * t102;
	t19 = k_fun_tmp * t104;
	t18 = b_fun_tmp * t103;
	t8 = c_fun_tmp * t104;
	t17 = t5 * t104;
	t7 = t5 * t103;
	t9 = t10 * t103 * t104;
	t6 = r_fun_tmp * t102 * t104;
	t10 = g_fun_tmp * t103 * t104;
	fun[3] = ((((((((((((((-in3[12] * (t14 * t23 * 3.0 - t15 * t23 * 15.0) -
							in3[8] * t146) -
							h_fun_tmp * t135_tmp) +
						i_fun_tmp * t23) +
						j_fun_tmp * t120) +
						t47 * t120) +
						t126_tmp * t120) +
					m_fun_tmp * t120 * (t117 - 1.0)) -
					n_fun_tmp * t146) -
					p_fun_tmp * (t135 + t137_tmp * 6.0)) -
					f_fun_tmp * t120 * q_fun_tmp) -
				t18 * t135_tmp) -
				t8 * t135_tmp) -
				l_fun_tmp * (t117 - 1.0) * t135_tmp) -
				e_fun_tmp * t139 * t135_tmp) +
			(((((((((t131_tmp * t23 + t131 * t23) + t19 * t23 * 6.0) +
					d_fun_tmp * t138 * t23) -
					o_fun_tmp * (t137 + b_t135_tmp * 6.0)) -
					t10 * t135_tmp) +
				t7 * t23) +
				t9 * t23 * 10.0) +
				t6 * t23 * 10.0) +
				t17 * t120);
	t5 = t132 - t141;
	fun[4] =
		(((((((((((((((((((((((-in3[12] * (t14 * t118 * 3.0 - t15 * t118 * 15.0) +
								in3[8] * t5) -
								j_fun_tmp * t108) +
							i_fun_tmp * t118) -
							h_fun_tmp * t123) -
							t47 * t108) -
							t126_tmp * t108) +
						t131_tmp * t118) +
						t131 * t118) +
						t19 * t118 * 6.0) -
						t18 * t123) -
					t8 * t123) -
					m_fun_tmp * t108 * (t117 - 1.0)) -
					l_fun_tmp * t123 * (t117 - 1.0)) +
					d_fun_tmp * t118 * t138) -
				e_fun_tmp * t123 * t139) +
				o_fun_tmp * (t132 - t141_tmp * 6.0)) -
				p_fun_tmp * (t141 - t132_tmp * 6.0)) +
				f_fun_tmp * t108 * q_fun_tmp) +
			n_fun_tmp * t5) -
			t17 * t108) +
			t7 * t118) +
			t9 * t118 * 10.0) +
		t6 * t118 * 10.0) -
		t10 * t123;
	fun[5] = (((((((((((((((((((((((in3[12] * (t14 * t107 * 3.0 -
												t16 * t107 * t110 * 15.0) +
									in3[8] * t148) -
									i_fun_tmp * t107) -
									j_fun_tmp * t121) +
								h_fun_tmp * t119) -
								t131_tmp * t107) -
								t131 * t107) -
								t19 * t107 * 6.0) +
							t18 * t119) -
							t47 * t121) -
							t126_tmp * t121) +
							t8 * t119) -
						m_fun_tmp * t121 * (t117 - 1.0)) +
						l_fun_tmp * t119 * (t117 - 1.0)) -
						d_fun_tmp * t107 * t138) +
						n_fun_tmp * t148) +
					e_fun_tmp * t119 * t139) +
					o_fun_tmp * (t140 + t136_tmp * 6.0)) +
					p_fun_tmp * (t136 + t140_tmp * 6.0)) +
					f_fun_tmp * t121 * q_fun_tmp) -
				t7 * t107) -
				t9 * t107 * 10.0) -
				t6 * t107 * 10.0) +
				t10 * t119) -
			t17 * t121;
}

}

double rcs_sph_gradient(
	Vec<3> const& radar_point,
	Vec<3> const& g_center,
	Vec<16> const& g_rcs,
	Vec<3> const& r_pos,
	Quat const& r_rot,
	Vec<6>& out_grad
)
{
	Vec<3> v = radar_point - r_rot.conjugate()*(g_center - r_pos);
	double vnorm = v.norm();
	if (v.x() > 0.0) {
		vnorm = -vnorm;
	}

	double rcs = make_sph<GaussianModel::G_SPH_LEVEL,double>(v/vnorm).dot(g_rcs);
	matlab_rcsSphGradient(radar_point.data(), g_center.data(), g_rcs.data(), r_pos.data(), r_rot.coeffs().data(), vnorm, out_grad.data());
	return rcs;
}

void GaussianModel::fit_rcs(
	RadarCloud const& cl,
	IncidenceCloud const& incicl,
	RcsParams const& p
)
{
	size_t numg = size();
	auto gm_match = matchup(cl);

	static constexpr double nan = std::numeric_limits<double>::quiet_NaN();
	rcs_scales.resize(Eigen::NoChange, numg);
	rcs_scales.fill(nan);
	rcs_coefs.resize(Eigen::NoChange, numg);
	rcs_coefs.fill(nan);

	struct Study {
		std::vector<uint32_t> children;
	};

	std::vector<Study> studies(numg);

	for (uint32_t i = 0; i < gm_match.size(); i++) {
		int32_t gidx = gm_match[i];
		if (gidx < 0) continue;
		studies[gidx].children.push_back(i);
	}

	for (size_t gidx = 0; gidx < numg; gidx ++) {
		Study& study = studies[gidx];
		if (study.children.size() < p.min_ppg) {
			continue;
		}

		printf("Gaussian %zu: %zu points\n", gidx, study.children.size());

		std::vector<int8_t> rcses(study.children.size());
		for (size_t j = 0; j < rcses.size(); j ++) {
			rcses[j] = cl[study.children[j]].power;
		}

		std::sort(rcses.begin(), rcses.end());

		int median = rcses[rcses.size()/2];

		int min = median - rcses[0];
		int max = rcses[rcses.size()-1] - median;
		int range = std::min(min, max);

		std::vector<Vec<G_SPH_NCOEFS,float>> shmtx;
		std::vector<float> rcstgt;

		for (uint32_t idx : study.children) {
			int rcslog = cl[idx].power;
			if (rcslog < median - range || rcslog > median + range) {
				continue;
			}

			Vecf<3> invec = incicl[idx].getVector3fMap() - incicl[idx].getLocalPose().inverse()*centers.col(gidx).cast<float>();
			//float rcs = std::pow(10.0, (float)(rcslog - median) / range);
			float rcs = (float)(rcslog - median) / range;

			shmtx.emplace_back(make_sph<G_SPH_LEVEL,float>(invec));
			rcstgt.emplace_back(rcs);
		}

		printf("  Ignored %zu points\n", rcses.size() - rcstgt.size());

		if (rcstgt.size() < p.min_ppg) {
			printf("  Noping out of this Gaussian\n");
			continue;
		}

		auto eig_shmtx = Eigen::Map<Eigen::MatrixXf>(shmtx.data()->data(), G_SPH_NCOEFS, shmtx.size()).transpose();
		auto eig_rcstgt = Eigen::Map<Eigen::VectorXf>(rcstgt.data(), rcstgt.size());

		auto solver = eig_shmtx.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
		if (solver.info() != Eigen::Success) {
			printf("  SVD returned Pifia\n");
			continue;
		}

		Vec<G_SPH_NCOEFS,float> gen_coefs = solver.solve(eig_rcstgt);
		//std::cout << "   coefs=[" << gen_coefs.transpose() << "]" << std::endl;

		Eigen::VectorXf rcs_error = eig_shmtx*gen_coefs - eig_rcstgt;
		double rcs_rmse = std::sqrt(rcs_error.array().square().mean());
		printf("   RCS RMSE = %f\n", rcs_rmse);
		//std::cout << "   [" << rcs_error.transpose() << "]" << std::endl;

		rcs_scales(0,gidx) = median;
		rcs_scales(1,gidx) = range;
		rcs_coefs.col(gidx) = gen_coefs;
	}

}

}
