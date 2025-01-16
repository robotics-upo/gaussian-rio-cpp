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

bool RioNdt::scan_matching(RadarCloud::Ptr cl)
{
	pcl::NormalDistributionsTransform<RadarPoint, RadarPoint> ndt;
	ndt.setTransformationEpsilon(m_ndt_params.xfrm_epsilon);
	ndt.setStepSize(m_ndt_params.step_size);
	ndt.setResolution(m_ndt_params.grid_resolution);
	ndt.setMaximumIterations(m_ndt_params.max_iters);

	ndt.setInputTarget(m_saved_cloud);
	ndt.setInputSource(cl);

	RadarCloud idc;
	ndt.align(idc, kf_pose().matrix().cast<float>());

	if (!ndt.hasConverged()) {
		std::cerr << "  [NDT] Convergence fail" << std::endl;
		return false;
	}

	auto xfrm = Pose(ndt.getFinalTransformation().cast<double>());
	update_scanmatch(xfrm);
	return true;
}

bool RioNdt::process_keyframe(RadarCloud::Ptr cl)
{
	m_saved_cloud = cl;
	return true;
}

}
