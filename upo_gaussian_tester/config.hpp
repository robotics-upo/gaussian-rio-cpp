#pragma once
#include <upo_gaussians/types.hpp>
#include <upo_gaussians/convention.hpp>
#include <upo_gaussians/rio_ndt.hpp>
#include <upo_gaussians/rio_gicp.hpp>
#include <upo_gaussians/rio_gaussian.hpp>
#include <yaml-cpp/yaml.h>
#include <variant>

namespace upo_gaussians {

	struct Dataset {

		struct ConfigCommon {
			std::string topic;
			Convention convention = Convention::NWU;
			Vec<3> pos = Vec<3>::Zero();
			Quat   rot = Quat::Identity();

			Pose pose() const { return make_pose(rot, pos); }
		};

		struct ImuConfig : ConfigCommon {
			double accel_std      = 0.01;
			double accel_bias_std = 0.0;
			double gyro_std       = 0.01;
			double gyro_bias_std  = 0.0;
		};

		struct RadarConfig : ConfigCommon {
			bool apply_filter         = false;
			std::string field_power   = "Power";
			std::string field_doppler = "Doppler";
		};

		struct Sequence {
			std::string name;
			std::string gt;
			std::vector<std::string> bags;
		};

		std::string path;
		std::string imu_name;
		std::string radar_name;
		ImuConfig imu;
		RadarConfig radar;
		std::vector<Sequence> seqs;
	};

	using RioConfig = std::variant<
		RioBase::InitParams,
		RioNdt::InitParams,
		RioGicp::InitParams,
		RioGaussian::InitParams
	>;

	namespace detail {

		template <typename T>
		struct RioConfigToMethod;

		template <> struct RioConfigToMethod<RioBase::InitParams>     { using Inner = RioBase;     };
		template <> struct RioConfigToMethod<RioNdt::InitParams>      { using Inner = RioNdt;      };
		template <> struct RioConfigToMethod<RioGicp::InitParams>     { using Inner = RioGicp;     };
		template <> struct RioConfigToMethod<RioGaussian::InitParams> { using Inner = RioGaussian; };

	}

	template <typename T>
	using RioConfigToMethod = typename detail::RioConfigToMethod<std::remove_cv_t<std::remove_reference_t<T>>>::Inner;

}

namespace YAML {

	template <typename T>
	static inline bool decode_in_place(T& out, Node const& node)
	{
		return convert<T>::decode(node, out);
	}

	template <typename T, int rows>
	struct convert<upo_gaussians::Vec<rows,T>> {
		static bool decode(Node const& node, upo_gaussians::Vec<rows,T>& rhs) {
			if (!node.IsSequence() || node.size() != rows) {
				return false;
			}

			for (Eigen::Index i = 0; i < rows; i ++) {
				rhs(i) = node[i].as<T>();
			}

			return true;
		}
	};

	template <typename T>
	struct convert<Eigen::Quaternion<T>> {
		static bool decode(Node const& node, Eigen::Quaternion<T>& rhs) {
			if (!node.IsSequence()) {
				return false;
			}

			if (node.size() == 4) {
				rhs.w() = node[0].as<T>();
				rhs.x() = node[1].as<T>();
				rhs.y() = node[2].as<T>();
				rhs.z() = node[3].as<T>();
			} else if (node.size() == 9) {
				upo_gaussians::Mat<3> m;
				for (size_t i = 0; i < 9; i ++) {
					m.data()[i] = node[i].as<T>();
				}
				rhs = m;
			} else {
				return false;
			}

			rhs.normalize();
			return true;
		}
	};

	template <>
	struct convert<upo_gaussians::Convention> {
		static bool decode(Node const& node, upo_gaussians::Convention& rhs) {
			auto str = node.as<std::string>();

			if (0) { }
#define _PAIN(_name) \
			else if (str == #_name) { rhs = upo_gaussians::Convention::_name; }
			_PAIN(NWU)
			_PAIN(NED)
			_PAIN(NDW)
			_PAIN(NUE)
			_PAIN(ENU)
			_PAIN(ESD)
			_PAIN(EUS)
			_PAIN(EDN)
#undef _PAIN
			else { return false; }

			return true;
		}
	};

#define _DECODE_FIELD_FULL(_node, _rhs, _field, _name) \
		(_rhs)._field = (_node)[_name].as<decltype((_rhs)._field)>()
#define _DECODE_FIELD_FULL_OPT(_node, _rhs, _field, _name) \
		do { if (auto _key = (_node)[_name]; _key) (_rhs)._field = _key.as<decltype((_rhs)._field)>(); } while (0)
#define _DECODE_FIELD(_field) _DECODE_FIELD_FULL(node, rhs, _field, #_field)
#define _DECODE_FIELD_OPT(_field) _DECODE_FIELD_FULL_OPT(node, rhs, _field, #_field)

	template <>
	struct convert<upo_gaussians::Dataset::ConfigCommon> {
		static bool decode(Node const& node, upo_gaussians::Dataset::ConfigCommon& rhs) {
			_DECODE_FIELD(topic);
			_DECODE_FIELD(convention);
			_DECODE_FIELD(pos);
			_DECODE_FIELD(rot);
			return true;
		}
	};

	template <>
	struct convert<upo_gaussians::Dataset::ImuConfig> {
		static bool decode(Node const& node, upo_gaussians::Dataset::ImuConfig& rhs) {
			if (!convert<upo_gaussians::Dataset::ConfigCommon>::decode(node, rhs)) return false;
			_DECODE_FIELD(accel_std);
			_DECODE_FIELD(accel_bias_std);
			_DECODE_FIELD(gyro_std);
			_DECODE_FIELD(gyro_bias_std);
			return true;
		}
	};

	template <>
	struct convert<upo_gaussians::Dataset::RadarConfig> {
		static bool decode(Node const& node, upo_gaussians::Dataset::RadarConfig& rhs) {
			if (!convert<upo_gaussians::Dataset::ConfigCommon>::decode(node, rhs)) return false;
			auto fields = node["fields"];
			_DECODE_FIELD_OPT(apply_filter);
			_DECODE_FIELD_FULL(fields, rhs, field_power, "power");
			_DECODE_FIELD_FULL(fields, rhs, field_doppler, "doppler");
			return true;
		}
	};

	template <>
	struct convert<upo_gaussians::Dataset::Sequence> {
		static bool decode(Node const& node, upo_gaussians::Dataset::Sequence& rhs) {
			_DECODE_FIELD(name);
			_DECODE_FIELD(gt);
			_DECODE_FIELD(bags);
			return true;
		}
	};

	template <>
	struct convert<upo_gaussians::Dataset> {
		static bool decode(Node const& node_top, upo_gaussians::Dataset& rhs) {
			auto node = node_top["dataset"];
			auto sensors = node_top["sensors"];
			_DECODE_FIELD(path);
			_DECODE_FIELD_FULL_OPT(node, rhs, imu_name, "imu");
			_DECODE_FIELD_FULL_OPT(node, rhs, radar_name, "radar");
			_DECODE_FIELD_FULL(sensors, rhs, imu, rhs.imu_name);
			_DECODE_FIELD_FULL(sensors, rhs, radar, rhs.radar_name);
			_DECODE_FIELD_FULL(node_top, rhs, seqs, "sequences");
			return true;
		}
	};

	template <>
	struct convert<upo_gaussians::RioBase::InitParams> {
		static bool decode(Node const& node, upo_gaussians::RioBase::InitParams& rhs) {
			_DECODE_FIELD_OPT(max_init_time);
			_DECODE_FIELD_OPT(r2i_tran_std);
			_DECODE_FIELD_OPT(r2i_rot_std);
			_DECODE_FIELD_OPT(gravity);
			_DECODE_FIELD_OPT(w_vel_std);
			_DECODE_FIELD_OPT(w_att_std);
			_DECODE_FIELD_OPT(voxel_size);
			_DECODE_FIELD_OPT(match_pos_std);
			_DECODE_FIELD_OPT(match_rot_std);
			_DECODE_FIELD_OPT(egovel_pct);
			_DECODE_FIELD_OPT(scanmatch_pct);
			_DECODE_FIELD_OPT(deterministic);
			_DECODE_FIELD_OPT(match_6dof);
			return true;
		}
	};

	template <>
	struct convert<upo_gaussians::RioNdt::NdtParams> {
		static bool decode(Node const& node, upo_gaussians::RioNdt::NdtParams& rhs) {
			_DECODE_FIELD_OPT(xfrm_epsilon);
			_DECODE_FIELD_OPT(step_size);
			_DECODE_FIELD_OPT(grid_resolution);
			_DECODE_FIELD_OPT(max_iters);
			return true;
		}
	};

	template <>
	struct convert<upo_gaussians::RioNdt::InitParams> {
		static bool decode(Node const& node, upo_gaussians::RioNdt::InitParams& rhs) {
			if (!convert<upo_gaussians::RioBase::InitParams>::decode(node, rhs)) return false;
			if (auto child = node["ndt"]; child) {
				if (!convert<upo_gaussians::RioNdt::NdtParams>::decode(child, rhs.ndt_params)) return false;
			}
			return true;
		}
	};

	template <>
	struct convert<upo_gaussians::RioGicp::GicpParams> {
		static bool decode(Node const& node, upo_gaussians::RioGicp::GicpParams& rhs) {
			_DECODE_FIELD_OPT(num_neighbors);
			_DECODE_FIELD_OPT(max_iters);
			_DECODE_FIELD_OPT(max_corr_dist);
			_DECODE_FIELD_OPT(tran_eps);
			_DECODE_FIELD_OPT(rot_eps);
			_DECODE_FIELD_OPT(vgicp_voxel);
			return true;
		}
	};

	template <>
	struct convert<upo_gaussians::RioGicp::InitParams> {
		static bool decode(Node const& node, upo_gaussians::RioGicp::InitParams& rhs) {
			if (!convert<upo_gaussians::RioBase::InitParams>::decode(node, rhs)) return false;
			if (auto child = node["gicp"]; child) {
				if (!convert<upo_gaussians::RioGicp::GicpParams>::decode(child, rhs.gicp_params)) return false;
			}
			return true;
		}
	};

	template <>
	struct convert<upo_gaussians::RioGaussian::InitParams> {
		static bool decode(Node const& node_top, upo_gaussians::RioGaussian::InitParams& rhs) {
			if (!convert<upo_gaussians::RioBase::InitParams>::decode(node_top, rhs)) return false;
			if (auto node = node_top["gaussian"]; node) {
				_DECODE_FIELD_OPT(gaussian_size);
				_DECODE_FIELD_OPT(num_particles);
				_DECODE_FIELD_OPT(match_thresh);
				_DECODE_FIELD_OPT(particle_std_xyz);
				_DECODE_FIELD_OPT(particle_std_rot);
			}
			return true;
		}
	};

	template <>
	struct convert<upo_gaussians::RioConfig> {
		static bool decode(Node const& node, upo_gaussians::RioConfig& rhs) {
			auto type = node["method"].as<std::string>();

			if (type == "no_scan_match") {
				rhs = node.as<upo_gaussians::RioBase::InitParams>();
			} else if (type == "ndt") {
				rhs = node.as<upo_gaussians::RioNdt::InitParams>();
			} else if (type == "gicp") {
				rhs = node.as<upo_gaussians::RioGicp::InitParams>();
			} else if (type == "gaussian") {
				rhs = node.as<upo_gaussians::RioGaussian::InitParams>();
			} else {
				return false;
			}

			return true;
		}
	};

#undef _DECODE_FIELD_FULL
#undef _DECODE_FIELD

}
