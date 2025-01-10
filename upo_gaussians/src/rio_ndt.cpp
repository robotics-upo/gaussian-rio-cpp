#include <upo_gaussians/rio_ndt.hpp>
#include <iostream>

#define PCL_NO_PRECOMPILE
#include <pcl/registration/ndt.h>

UPO_GAUSSIANS_REGISTER_RADAR_POINT_STRUCT(Power, Doppler)

namespace upo_gaussians {

RioNdt::RioNdt(
	Keyframer keyframer,
	InitParams const& p
) : RioBase{keyframer,p}, m_ndt_params{p.ndt_params}
{
}

void RioNdt::scan_matching(RadarCloud const& cl)
{
	pcl::NormalDistributionsTransform<RadarPoint, RadarPoint> ndt;
	ndt.setTransformationEpsilon(m_ndt_params.xfrm_epsilon);
	ndt.setStepSize(m_ndt_params.step_size);
	ndt.setResolution(m_ndt_params.grid_resolution);
	ndt.setMaximumIterations(m_ndt_params.max_iters);

	ndt.setInputTarget(m_saved_cloud);
	ndt.setInputSource(std::make_shared<RadarCloud>(cl)); // PCL is stupid, and forces us to copy this

	RadarCloud idc;
	ndt.align(idc, kf_pose().matrix().cast<float>());

	if (!ndt.hasConverged()) {
		std::cerr << "  [NDT] Convergence fail" << std::endl;
		return;
	}

	auto xfrm = Pose(ndt.getFinalTransformation().cast<double>());
	update_scanmatch_3d(xfrm);
}

bool RioNdt::process_keyframe(RadarCloud&& cl)
{
	m_saved_cloud.reset();
	m_saved_cloud = std::make_shared<RadarCloud>(std::move(cl));

	return true;
}

}
