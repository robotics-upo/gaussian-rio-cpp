#include <upo_gaussians/kalman_core.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <iostream>

namespace upo_gaussians {

Strapdown::Strapdown(
	InitParams const& p,
	bool want_yaw_gyro_bias
)
{
	int z = want_yaw_gyro_bias ? 3 : 2;
	m_cov.block(AccBias, AccBias, 3, 3).diagonal().fill(p.accel_bias_std*p.accel_bias_std);
	m_cov.block(GyrBias, GyrBias, z, z).diagonal().fill(p.gyro_bias_std*p.gyro_bias_std);
	m_cov.block(AttError, AttError, 2, 2).diagonal().fill(p.rp_att_std*p.rp_att_std);
}

void Strapdown::init_vel(
	Vec<3> const& vel,
	Mat<3> const& vel_cov
)
{
	m_state.segment<3>(Vel) = vel;
	m_cov.block(Vel, Vel, 3, 3) = vel_cov;
}

void Strapdown::propagate_imu(
	double timediff,
	Vec<3> const& accel,
	Vec<3> const& accel_covdiag,
	Vec<3> const& gyro,
	Vec<3> const& gyro_covdiag,
	PropParams const& p
)
{
	auto R = m_attitude.matrix();
	auto accel_world = m_attitude * (accel - accel_bias());
	auto A = skewsym(accel_world);
	auto g = Vec<3>(0, 0, -p.gravity);

	m_state.segment<3>(Pos) += timediff*velocity() + 0.5*timediff*timediff*(accel_world + g);
	m_state.segment<3>(Vel) += timediff*(accel_world + g);
	m_attitude = m_attitude * so3_exp(timediff*(gyro - gyro_bias()));

	Mat<CovTotal> F;
	F.setIdentity();
	F.block(Pos,Vel,3,3).diagonal().fill(timediff);
	F.block(Pos,AccBias,3,3) = -0.5*timediff*timediff*R;
	F.block(Pos,AttError,3,3) = -0.5*timediff*timediff*A;
	F.block(Vel,AccBias,3,3) = F.block(AttError,GyrBias,3,3) = -timediff*R;
	F.block(Vel,AttError,3,3) = -timediff*A;

	Vec<WTotal> Qdiag;
	Qdiag.segment<3>(WVel).fill(p.w_vel_std*p.w_vel_std);
	Qdiag.segment<3>(WAtt).fill(p.w_att_std*p.w_att_std);
	Qdiag.segment<3>(WAcc) = accel_covdiag / timediff;
	Qdiag.segment<3>(WGyr) = gyro_covdiag / timediff;
	Qdiag.segment<3>(WAccBias).fill(timediff*p.w_accel_bias_std*p.w_accel_bias_std);
	Qdiag.segment<3>(WGyrBias).fill(timediff*p.w_gyro_bias_std*p.w_gyro_bias_std);

	Mat<CovTotal,WTotal> N;
	N.setZero();
	N.block(Pos,WAcc,3,3) = 0.5*timediff*timediff*R;
	N.block(Vel,WAcc,3,3) = N.block(AttError,WGyr,3,3) = timediff*R;
	N.block(Vel,WVel,3,3) = N.block(AccBias,WAccBias,3,3) = N.block(GyrBias,WGyrBias,3,3) = N.block(AttError,WAtt,3,3) = Mat<3>::Identity();

	m_cov = F*m_cov*F.transpose() + N*Qdiag.asDiagonal()*N.transpose();
}

template <int N, typename RType>
void Strapdown::update_common(
	const char* name,
	Vec<N> const& residual,
	RType const& R,
	Mat<N,CovTotal> const& H,
	double outlier_percentile
)
{
	auto Sinv = (H*m_cov*H.transpose() + as_dense(R)).ldlt();

	if (outlier_percentile > 0.0) {
		double gamma = residual.transpose() * Sinv.solve(residual);
		double gamma_thresh = boost::math::quantile(boost::math::chi_squared{N}, 1.0 - outlier_percentile);

		if (gamma >= gamma_thresh) {
			std::cout << " [EKF] " << name << " update rejected, gamma=" << gamma << " >= " << gamma_thresh << std::endl;
			return;
		}
	}

	auto K = Sinv.solve((m_cov*H.transpose()).transpose()).transpose().eval();
	auto L = (Cov::Identity() - K*H).eval();

	auto gain = (K*residual).eval();
	m_state += gain.template segment<StateTotal>(0);
	m_cov = L*m_cov*L.transpose() + K*R*K.transpose();

	// ESKF reset
	auto qerr = so3_exp(gain.template segment<3>(AttError));
	Mat<CovTotal> G;
	G.setIdentity();
	G.block(AttError,AttError,3,3) = qerr.matrix();
	m_attitude = qerr*m_attitude;
	m_attitude.normalize(); // renormalize to avoid rounding errors
	m_cov = G*m_cov*G.transpose();
}

Vec<3> Strapdown::calc_egovel(
	Vec<3> const& angvel,
	Pose const& radar_to_imu
) const
{
	return radar_to_imu.rotation().transpose()*(angvel.cross(radar_to_imu.translation()) + m_attitude.conjugate()*velocity());
}

void Strapdown::update_egovel(
	Vec<3> const& egovel,
	Mat<3> const& egovel_cov,
	Vec<3> const& angvel,
	Pose   const& radar_to_imu,
	double outlier_percentile
)
{
	auto Cwb = m_attitude.conjugate().matrix();
	auto Cbr = radar_to_imu.rotation().transpose();
	auto pred_egovel = calc_egovel(angvel, radar_to_imu);

	Mat<3,CovTotal> H;
	H.setZero();
	H.block(0,Vel,3,3) = Cbr*Cwb;
	H.block(0,GyrBias,3,3) = Cbr*skewsym(radar_to_imu.translation());
	H.block(0,AttError,3,3) = Cbr*Cwb*skewsym(velocity());

	update_common("egovel", (egovel - pred_egovel).eval(), egovel_cov, H, outlier_percentile);
}

void Strapdown::update_scanmatch(
	Pose const& kf_pose,
	Mat<6> const& kf_cov,
	Pose const& match_pose,
	Vec<6> const& match_covdiag,
	bool full_6dof,
	double outlier_percentile
)
{
	Pose pred_pose = kf_pose.inverse() * pose();
	Quat diff_q{ pred_pose.rotation().transpose() * match_pose.rotation() };

	Vec<6> residual;
	residual.segment<3>(0) = match_pose.translation() - pred_pose.translation();
	residual.segment<3>(3) = so3_log(diff_q);

	Mat<6,CovTotal> H;
	H.setZero();
	H.block(0,Pos,3,3) = H.block(3,AttError,3,3) = kf_pose.rotation().transpose();

	Mat<6> R = kf_cov + match_covdiag.asDiagonal().toDenseMatrix();

	if (full_6dof) {
		update_common("scanmatch_6dof", residual, R, H, outlier_percentile);
	} else {
		Mat<3,6> Hc;
		Hc.setZero();
		Hc(0,0) = Hc(1,1) = Hc(2,5) = 1;

		Vec<3> newresidual = Hc*residual;
		Mat<3> newR = Hc*R*Hc.transpose();
		Mat<3,CovTotal> newH = Hc*H;

		update_common("scanmatch_3dof", newresidual, newR, newH, outlier_percentile);
	}
}

Mat<6> Strapdown::error_cov() const
{
	Mat<6,CovTotal> H;
	H.setZero();
	H.block(0,Pos,3,3) = H.block(3,AttError,3,3) = m_attitude.conjugate().matrix();

	return H*m_cov*H.transpose();
}

}
