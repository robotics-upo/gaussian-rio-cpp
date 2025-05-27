#include <stdio.h>
#include <getopt.h>
#include <sys/sysinfo.h>

#include <iostream>
#include <fstream>

#include "config.hpp"
#include "ros_utils.hpp"

namespace {
	using namespace upo_gaussians;

	bool is_keyframe(RioBase const& rio)
	{
		if (!rio.has_keyframe()) {
			return true;
		}

		if (rio.match_time() >= 0.5) {
			std::cout << "  !!!!! too long since last match" << std::endl;
			return true;
		}

		auto pose = rio.kf_pose();
		double xlate = pose.translation().head<2>().norm();
		Quat q{pose.rotation()};
		double rot = 2.0*acos(fabs(q.w()));

		return xlate >= 15.0 || rot >= 5.0*M_TAU/360.0;
	}

	Vec<3> quat_to_rpy(Quat const& q)
	{
		auto qw = q.w(), qx = q.x(), qy = q.y(), qz = q.z();

		// roll (x-axis rotation)
		auto sinr_cosp = 2*(qw*qx + qy*qz);
		auto cosr_cosp = 1 - 2*(qx*qx + qy*qy);
		auto roll = atan2(sinr_cosp, cosr_cosp);

		// pitch (y-axis rotation)
		auto sinp = sqrt(1 + 2*(qw*qy - qx*qz));
		auto cosp = sqrt(1 - 2*(qw*qy - qx*qz));
		auto pitch = 2*atan2(sinp, cosp) - M_TAU/4;

		// yaw (z-axis rotation)
		auto siny_cosp = 2*(qw*qz + qx*qy);
		auto cosy_cosp = 1 - 2*(qy*qy + qz*qz);
		auto yaw = atan2(siny_cosp, cosy_cosp);

		return Vec<3>(roll, pitch, yaw);
	}

	void run_odometry(
		RioBase& rio,
		Dataset const& ds,
		Dataset::Sequence const& seq,
		std::string const& outname
	)
	{
		std::cout << "-------- Sequence " << seq.name << " --------" << std::endl;

		RioBase::Input rio_input;
		unsigned scan_id = 0;

		ImuData::DecodeParams imu_params;
		imu_params.accel_std  = ds.imu.accel_std;
		imu_params.gyro_std   = ds.imu.gyro_std;
		imu_params.convention = ds.imu.convention;

		FILE *fout_traj, *fout_extra;
		{
			std::string traj_fname  = outname + "_traj_"  + seq.name + ".txt";
			std::string extra_fname = outname + "_extra_" + seq.name + ".txt";
			fout_traj  = fopen(traj_fname.c_str(),  "w");
			fout_extra = fopen(extra_fname.c_str(), "w");
		}

		if (fout_traj) {
			fprintf(fout_traj, "# timestamp tx ty tz qx qy qz qw\n");
		}

		for (auto& bagname : seq.bags) {
			rosbag::Bag bag(ds.path + "/" + bagname);

			for (auto msg : rosbag::View(bag)) {
				auto topic = msg.getTopic();

				if (topic == ds.imu.topic) {
					auto pimu = msg.instantiate<sensor_msgs::Imu>();
					if (pimu) {
						auto& imu = rio_input.imu_data.emplace_back(ImuData::fromROS(*pimu, imu_params));
						imu.accel = ds.imu.rot*imu.accel;
						imu.gyro = ds.imu.rot*imu.gyro;
					}
				} else if (topic == ds.radar.topic) {
					if (!rosbag_msg_to_radar_cloud(rio_input.radar_scan, rio_input.radar_time, msg, ds.radar)) {
						continue;
					}

					printf("[Scan %5u] Radar data at %f (%zu points) with %zu IMU messages\n",
						scan_id++, rio_input.radar_time, rio_input.radar_scan.size(), rio_input.imu_data.size());
					rio.process(rio_input);
					rio_input.imu_data.clear();

					if (rio.is_initial()) {
						continue;
					}

					auto t = rio.position();
					auto q = rio.attitude();
					auto ab = rio.accel_bias();
					auto wb = rio.gyro_bias();
					auto ev = rio.egovel();
					auto r2it = rio.r2i_tran();
					auto r2iq = rio.r2i_rot();

					std::cout << "  Pose: t=[" << t.transpose() << "]; rot=[" << quat_to_rpy(q).transpose()*360.0/M_TAU << "]" << std::endl;

					if (fout_traj) {
						fprintf(fout_traj, "%f %.15g %.15g %.15g %.15g %.15g %.15g %.15g\n",
							rio_input.radar_time,
							t.x(), t.y(), t.z(),
							q.x(), q.y(), q.z(), q.w()
						);
					}

					if (fout_extra) {
						fprintf(fout_extra,
							"%.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g\n",
							ev(0), ev(1), ev(2),
							ab(0), ab(1), ab(2),
							wb(0), wb(1), wb(2),
							r2it(0), r2it(1), r2it(2),
							r2iq.x(), r2iq.y(), r2iq.z(), r2iq.w()
						);
					}
				}
			}

		}

		if (fout_traj)  fclose(fout_traj);
		if (fout_extra) fclose(fout_extra);
	}

	const option long_options[] = {
		{ "output",  required_argument, nullptr, 'o' },
		{ "dataset", required_argument, nullptr, 'd' },
		{ "config",  required_argument, nullptr, 'c' },
		{ "radar",   required_argument, nullptr, 'r' },
		{ "imu",     required_argument, nullptr, 'i' },
		{ },
	};

	int usage(const char* progname)
	{
		fprintf(stderr,
			"Usage:\n  %s "
			"[--output name] "
			"--dataset file.yaml "
			"--config config.yaml "
			"[--radar radarname] "
			"[--imu imuname]\n",
			progname);
		return EXIT_FAILURE;
	}
}

int main(int argc, char* argv[])
{
	const char* p_output  = "./output";
	const char* p_dataset = nullptr;
	const char* p_config  = nullptr;
	const char* p_radar   = "radar";
	const char* p_imu     = "imu";

	int opt, optidx = 0;
	while ((opt = getopt_long(argc, argv, "odcri", long_options, &optidx)) != -1) {
		switch (opt) {
			case 'o': p_output  = optarg; break;
			case 'd': p_dataset = optarg; break;
			case 'c': p_config  = optarg; break;
			case 'r': p_radar   = optarg; break;
			case 'i': p_imu     = optarg; break;
			default:  return usage(argv[0]);
		}
	}

	if (!p_dataset) {
		fprintf(stderr, "Missing --dataset argument\n");
		return usage(argv[0]);
	}

	if (!p_config) {
		fprintf(stderr, "Missing --config argument\n");
		return usage(argv[0]);
	}

	if (optind > argc) {
		fprintf(stderr, "Extraneous arguments passed\n");
		return usage(argv[0]);
	}

	Dataset ds;
	ds.radar_name = p_radar;
	ds.imu_name   = p_imu;
	if (!YAML::decode_in_place(ds, YAML::LoadFile(p_dataset))) {
		fprintf(stderr, "Unable to parse dataset config\n");
		return EXIT_FAILURE;
	}

	RioConfig cfg;
	if (!YAML::decode_in_place(cfg, YAML::LoadFile(p_config))) {
		fprintf(stderr, "Unable to parse RIO config\n");
		return EXIT_FAILURE;
	}

	std::visit([&](auto& rio_params) {
		rio_params.num_threads = get_nprocs();
		rio_params.w_accel_bias_std = ds.imu.accel_bias_std;
		rio_params.w_gyro_bias_std  = ds.imu.gyro_bias_std;
		rio_params.radar_to_imu = make_pose(Quat::Identity(), ds.imu.pos).inverse()*ds.radar.pose();
		rio_params.filter_cloud = ds.radar.apply_filter;

		for (auto& seq : ds.seqs) {
			RioConfigToMethod<decltype(rio_params)> rio{is_keyframe, rio_params};
			run_odometry(rio, ds, seq, p_output);
		}
	}, cfg);

	return EXIT_SUCCESS;
}
