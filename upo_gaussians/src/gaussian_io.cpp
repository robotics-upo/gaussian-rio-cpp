#include <upo_gaussians/gaussian_model.hpp>

namespace upo_gaussians {

namespace {

	static constexpr uint32_t GAUSSIAN_MAGIC = 0xfafafafa;
	static constexpr uint32_t GFLAG_HAS_SPH  = 1U<<0;

	struct GaussianFileHdr {
		uint32_t magic;
		uint32_t num_gaussians;
		uint32_t flags;
	};

}

void GaussianModel::save(const char* fname) const
{
	FILE* f = fopen(fname, "wb");
	if (!f) {
		fprintf(stderr, "Cannot open %s for writing\n", fname);
		std::abort();
	}

	// Write header
	{
		GaussianFileHdr hdr = {
			.magic         = GAUSSIAN_MAGIC,
			.num_gaussians = (uint32_t)size(),
			.flags         = (has_rcs() ? GFLAG_HAS_SPH : 0),
		};

		fwrite(&hdr, sizeof(hdr), 1, f);
	}

	// Write main Gaussian parameters
	fwrite(centers.data(), sizeof(*centers.data()), centers.size(), f);
	fwrite(log_scales.data(), sizeof(*log_scales.data()), log_scales.size(), f);
	fwrite(quats.data(), sizeof(*quats.data()), quats.size(), f);

	// Write RCS
	if (has_rcs()) {
		fwrite(rcs_scales.data(), sizeof(*rcs_scales.data()), rcs_scales.size(), f);
		fwrite(rcs_coefs.data(), sizeof(*rcs_coefs.data()), rcs_coefs.size(), f);
	}

	fclose(f);
}

bool GaussianModel::load(const char* fname)
{
	FILE* f = fopen(fname, "rb");
	if (!f) {
		fprintf(stderr, "Cannot open %s for reading\n", fname);
		std::abort();
	}

	#define chk_read(_a,_b,_c) do { size_t __c = (_c); if (fread((_a), (_b), __c, f) != __c) { fclose(f); return false; } } while (0)

	GaussianFileHdr hdr;
	chk_read(&hdr, sizeof(hdr), 1);
	if (hdr.magic != GAUSSIAN_MAGIC || !hdr.num_gaussians) {
		fclose(f);
		return false;
	}

	centers.resize(Eigen::NoChange, hdr.num_gaussians);
	log_scales.resize(Eigen::NoChange, hdr.num_gaussians);
	quats.resize(hdr.num_gaussians);

	if (hdr.flags & GFLAG_HAS_SPH) {
		rcs_scales.resize(hdr.num_gaussians);
		rcs_coefs.resize(Eigen::NoChange, hdr.num_gaussians);
	}

	chk_read(centers.data(), sizeof(*centers.data()), centers.size());
	chk_read(log_scales.data(), sizeof(*log_scales.data()), log_scales.size());
	chk_read(quats.data(), sizeof(*quats.data()), quats.size());

	if (hdr.flags & GFLAG_HAS_SPH) {
		chk_read(rcs_scales.data(), sizeof(*rcs_scales.data()), rcs_scales.size());
		chk_read(rcs_coefs.data(), sizeof(*rcs_coefs.data()), rcs_coefs.size());
	}

	fclose(f);
	return true;
}

}
