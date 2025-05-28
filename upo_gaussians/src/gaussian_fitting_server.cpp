#include <upo_gaussians/gaussian_model.hpp>

namespace upo_gaussians {

void GaussianModel::fit_server(AnyCloudIn cl, FitParams const& p)
{
	FILE* fout = fopen("/tmp/gaussian_server_in", "wb");
	if (!fout) {
		fprintf(stderr, "Gaussian server not running\n");
		std::abort();
	}

	FILE* fin = fopen("/tmp/gaussian_server_out", "rb");
	if (!fin) {
		fclose(fout);
		fprintf(stderr, "Gaussian server not running\n");
		std::abort();
	}

	centers.resize(Eigen::NoChange, p.num_gaussians);
	log_scales.resize(Eigen::NoChange, p.num_gaussians);
	quats.resize(p.num_gaussians);

	uint32_t cmd = 0;
	fwrite(&cmd, sizeof(cmd), 1, fout);
	fwrite(&p, sizeof(p), 1, fout);
	uint32_t num_points = cl.cols();
	fwrite(&num_points, sizeof(num_points), 1, fout);

	for (Eigen::Index i = 0; i < cl.cols(); i ++) {
		fwrite(cl.col(i).data(), sizeof(float), 3, fout);
	}

	fclose(fout);

	size_t cr = fread(centers.data(), sizeof(*centers.data()), centers.size(), fin);
	size_t sr = fread(log_scales.data(), sizeof(*log_scales.data()), log_scales.size(), fin);
	size_t qr = fread(quats.data(), sizeof(*quats.data()), quats.size(), fin);

	fclose(fin);

	if (cr != centers.size() || sr != log_scales.size() || qr != quats.size()) {
		fprintf(stderr, "Gaussian server did not output enough data\n");
		fprintf(stderr, "%zu %zu %zu\n", cr, sr, qr);
		std::abort();
	}
}

}
