#include <stdio.h>
#include <getopt.h>
#include <sys/sysinfo.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <csignal>

#include "config.hpp"
#include "ros_utils.hpp"

#include <tf2_msgs/TFMessage.h>
#include <geometry_msgs/TransformStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Odometry.h>

namespace {
	using namespace upo_gaussians;

	volatile bool has_ctrlc = false;

	void sigint_handler(int signum)
	{
		signal(SIGINT, sigint_handler);
		if (!has_ctrlc) {
			printf("Interrupted\n");
			has_ctrlc = true;
		}
	}

	void setup_ctrlc()
	{
		signal(SIGINT, sigint_handler);
	}

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
		std::string dump_fname = outname + "_dumps_" + seq.name + ".bag";
		{
			std::string traj_fname  = outname + "_traj_"  + seq.name + ".txt";
			std::string extra_fname = outname + "_extra_" + seq.name + ".txt";

			fout_traj  = fopen(traj_fname.c_str(),  "w");
			fout_extra = fopen(extra_fname.c_str(), "w");
		}

		if (fout_traj) {
			fprintf(fout_traj, "# timestamp tx ty tz qx qy qz qw\n");
		}

		rosbag::Bag dumpbag(dump_fname, rosbag::bagmode::Write);

		geometry_msgs::TransformStamped keyframe;
		keyframe.header.frame_id = "map";
		keyframe.child_frame_id = "keyframe";
		keyframe.transform.rotation.w = 1.0;

		nav_msgs::Odometry navmsg;
		navmsg.header.frame_id = "map";
		navmsg.child_frame_id = "odom";

		RioGaussian* grio = dynamic_cast<RioGaussian*>(&rio);

		bool did_kf_cloud = false;

		for (auto& bagname : seq.bags) {
			if (has_ctrlc) break;

			rosbag::Bag bag(ds.path + "/" + bagname);

			for (auto msg : rosbag::View(bag)) {
				if (has_ctrlc) break;

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

					unsigned cur_scan_id = scan_id++;
					printf("[Scan %5u] Radar data at %f (%zu points) with %zu IMU messages\n",
						cur_scan_id, rio_input.radar_time, rio_input.radar_scan.size(), rio_input.imu_data.size());

					auto tref = std::chrono::steady_clock::now();
					rio.process(rio_input);
					double process_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tref).count();

					rio_input.imu_data.clear();

					auto t = rio.position();
					auto q = rio.attitude();
					auto ab = rio.accel_bias();
					auto wb = rio.gyro_bias();
					auto ev = rio.egovel();
					auto r2it = rio.r2i_tran();
					auto r2iq = rio.r2i_rot();

					auto kfrelpose = rio.kf_pose();
					auto kfrelt = Vec<3>{kfrelpose.translation()};
					auto kfrelq = Quat{kfrelpose.rotation()};

					if (!rio.is_initial()) {
						std::cout << "  Process time: " << process_ms << " ms" << std::endl;
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
								"%u %u %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g\n",
								cur_scan_id, rio.at_keyframe(), process_ms,
								ev(0), ev(1), ev(2),
								ab(0), ab(1), ab(2),
								wb(0), wb(1), wb(2),
								r2it(0), r2it(1), r2it(2),
								r2iq.x(), r2iq.y(), r2iq.z(), r2iq.w(),
								kfrelt(0), kfrelt(1), kfrelt(2),
								kfrelq.x(), kfrelq.y(), kfrelq.z(), kfrelq.w()
							);
						}
					}

					auto rosnow = ros::Time(rio_input.radar_time);

					// Write odometry
					{
						tf2_msgs::TFMessage tf;

						if (!rio.is_initial()) {
							geometry_msgs::TransformStamped odom;
							odom.header.stamp = rosnow;
							odom.header.frame_id = "keyframe";
							odom.child_frame_id = "odom";
							odom.transform.translation.x = kfrelt(0);
							odom.transform.translation.y = kfrelt(1);
							odom.transform.translation.z = kfrelt(2);
							odom.transform.rotation.x = kfrelq.x();
							odom.transform.rotation.y = kfrelq.y();
							odom.transform.rotation.z = kfrelq.z();
							odom.transform.rotation.w = kfrelq.w();
							tf.transforms.emplace_back(std::move(odom));

							navmsg.header.stamp = rosnow;
							navmsg.pose.pose.position.x = t(0);
							navmsg.pose.pose.position.y = t(1);
							navmsg.pose.pose.position.z = t(2);
							navmsg.pose.pose.orientation.x = q.x();
							navmsg.pose.pose.orientation.y = q.y();
							navmsg.pose.pose.orientation.z = q.z();
							navmsg.pose.pose.orientation.w = q.w();

							Mat<6> posecov = rio.pose_cov();
							memcpy(navmsg.pose.covariance.data(), posecov.data(), sizeof(navmsg.pose.covariance));

							Vec<3> vel = rio.attitude().conjugate()*rio.velocity();
							Vec<3> angvel = rio.angvel();
							navmsg.twist.twist.linear.x = vel(0);
							navmsg.twist.twist.linear.y = vel(1);
							navmsg.twist.twist.linear.z = vel(2);
							navmsg.twist.twist.angular.x = angvel(0);
							navmsg.twist.twist.angular.y = angvel(1);
							navmsg.twist.twist.angular.z = angvel(2);

							dumpbag.write("/grio/odom", rosnow, navmsg);
						}

						if (rio.at_keyframe()) {
							keyframe.transform.translation.x = t(0);
							keyframe.transform.translation.y = t(1);
							keyframe.transform.translation.z = t(2);
							keyframe.transform.rotation.x = q.x();
							keyframe.transform.rotation.y = q.y();
							keyframe.transform.rotation.z = q.z();
							keyframe.transform.rotation.w = q.w();
						}

						keyframe.header.stamp = rosnow;
						tf.transforms.push_back(keyframe);

						dumpbag.write("/tf", rosnow, tf);
					}

					// Write the pointcloud to a message
					{
						sensor_msgs::PointCloud2 roscl;
						pcl::toROSMsg(*rio.last_cloud(), roscl);

						roscl.header.stamp = rosnow;
						roscl.header.frame_id = "odom";
						dumpbag.write("/grio/processed_radar", rosnow, roscl);
					}

					// Write keyframe pointcloud to a message
					if (!rio.is_initial() && (!did_kf_cloud || rio.at_keyframe())) {
						did_kf_cloud = true;

						sensor_msgs::PointCloud2 roscl;
						pcl::toROSMsg(*rio.kf_cloud(), roscl);

						roscl.header.stamp = rosnow;
						roscl.header.frame_id = "keyframe";
						dumpbag.write("/grio/keyframe_cloud", rosnow, roscl);
					}

					// Write the Gaussians
					if (grio && grio->model().size()) {
						auto& model = grio->model();

						visualization_msgs::MarkerArray marr;
						{
							visualization_msgs::Marker m;
							m.header.stamp = rosnow;
							m.header.frame_id = "keyframe";
							m.action = visualization_msgs::Marker::DELETEALL;
							marr.markers.emplace_back(std::move(m));
						}

						for (Eigen::Index i = 0; i < model.size(); i ++) {
							visualization_msgs::Marker m;
							m.header.stamp = rosnow;
							m.header.frame_id = "keyframe";
							m.action = visualization_msgs::Marker::ADD;
							m.id = i;
							m.type = visualization_msgs::Marker::SPHERE;
							m.pose.position.x = model.centers.col(i)(0);
							m.pose.position.y = model.centers.col(i)(1);
							m.pose.position.z = model.centers.col(i)(2);
							m.pose.orientation.x = model.quats(i).x();
							m.pose.orientation.y = model.quats(i).y();
							m.pose.orientation.z = model.quats(i).z();
							m.pose.orientation.w = model.quats(i).w();
							Vec<3> scale = 2.0*model.log_scales.col(i).array().exp().matrix();
							m.scale.x = scale(0);
							m.scale.y = scale(1);
							m.scale.z = scale(2);
							m.color.a = 0.5;
							m.color.r = 0.0;
							m.color.g = 1.0;
							m.color.b = 0.0;
							marr.markers.emplace_back(std::move(m));
						}

						dumpbag.write("/grio/gaussians", rosnow, marr);
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

	setup_ctrlc();

	std::visit([&](auto& rio_params) {
		rio_params.num_threads = get_nprocs();
		rio_params.w_accel_bias_std = ds.imu.accel_bias_std;
		rio_params.w_gyro_bias_std  = ds.imu.gyro_bias_std;
		rio_params.radar_to_imu = make_pose(Quat::Identity(), ds.imu.pos).inverse()*ds.radar.pose();
		rio_params.filter_cloud = ds.radar.apply_filter;

		for (auto& seq : ds.seqs) {
			if (has_ctrlc) break;

			RioConfigToMethod<decltype(rio_params)> rio{is_keyframe, rio_params};
			run_odometry(rio, ds, seq, p_output);
		}
	}, cfg);

	return EXIT_SUCCESS;
}
