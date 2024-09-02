#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <mcmcsv.h>

class McmcSv {
public:
	McmcSv(
		int num_chains, int num_iter, int num_burn, int thin,
		const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		py::dict& param_sv, py::dict& param_prior, py::dict& param_intercept,
		std::vector<py::dict>& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::MatrixXi& grp_mat,
		bool include_mean, const Eigen::VectorXi& seed_chain,
		bool display_progress, int nthreads
	)
	: num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin), nthreads(nthreads),
		display_progress(display_progress), sv_objs(num_chains), res(num_chains) {
		switch (prior_type) {
			case 1: {
				bvhar::MinnSvParams minn_params(
					num_iter, x, y,
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int i = 0; i < num_chains; i++ ) {
					bvhar::SvInits sv_inits(param_init[i]);
					sv_objs[i].reset(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 2: {
				bvhar::SsvsSvParams ssvs_params(
					num_iter, x, y,
					param_sv,
					grp_id, grp_mat,
					param_prior,
					param_intercept,
					include_mean
				);
				for (int i = 0; i < num_chains; i++ ) {
					bvhar::SsvsSvInits ssvs_inits(param_init[i]);
					sv_objs[i].reset(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 3: {
				bvhar::HsSvParams horseshoe_params(
					num_iter, x, y,
					param_sv,
					grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int i = 0; i < num_chains; i++ ) {
					bvhar::HsSvInits hs_inits(param_init[i]);
					sv_objs[i].reset(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 4: {
				bvhar::HierminnSvParams minn_params(
					num_iter, x, y,
					param_sv,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int i = 0; i < num_chains; i++ ) {
					bvhar::HierminnSvInits minn_inits(param_init[i]);
					sv_objs[i].reset(new bvhar::HierminnSv(minn_params, minn_inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 5: {
				bvhar::NgSvParams ng_params(
					num_iter, x, y,
					param_sv,
					grp_id, grp_mat,
					param_prior,
					param_intercept,
					include_mean
				);
				for (int i = 0; i < num_chains; ++i) {
					bvhar::NgSvInits ng_inits(param_init[i]);
					sv_objs[i].reset(new bvhar::NormalgammaSv(ng_params, ng_inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 6: {
				bvhar::DlSvParams dl_params(
					num_iter, x, y,
					param_sv,
					grp_id, grp_mat,
					param_prior,
					param_intercept,
					include_mean
				);
				for (int i = 0; i < num_chains; ++i) {
					bvhar::GlSvInits dl_inits(param_init[i]);
					sv_objs[i].reset(new bvhar::DirLaplaceSv(dl_params, dl_inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
		}
	}
	virtual ~McmcSv() = default;
	std::vector<py::dict> returnRecords() {
		fit();
		return res;
	}

protected:
	void runGibbs(int chain) {
		bvhar::bvharprogress bar(num_iter, display_progress);
		for (int i = 0; i < num_iter; ++i) {
			bar.increment();
			sv_objs[chain]->doPosteriorDraws();
			bar.update();
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = sv_objs[chain]->returnRecords(num_burn, thin);
		}
	}
	void fit() {
		if (num_chains == 1) {
			runGibbs(0);
		} else {
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads)
		#endif
			for (int chain = 0; chain < num_chains; chain++) {
				runGibbs(chain);
			}
		}
	}

private:
	int num_chains;
	int num_iter;
	int num_burn;
	int thin;
	int nthreads;
	bool display_progress;
	std::vector<std::unique_ptr<bvhar::McmcSv>> sv_objs;
	std::vector<py::dict> res;
};

PYBIND11_MODULE(_sv, m) {
	py::class_<McmcSv>(m, "McmcSv")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&, py::dict&,
			std::vector<py::dict>&, int, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &McmcSv::returnRecords);
}