// #include <mcmcsv.h>
#include <bvharmcmc.h>

// class SvMcmc {
// public:
// 	SvMcmc(
// 		int num_chains, int num_iter, int num_burn, int thin,
// 		const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
// 		py::dict& param_sv, py::dict& param_prior, py::dict& param_intercept,
// 		std::vector<py::dict>& param_init, int prior_type,
// 		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
// 		const Eigen::MatrixXi& grp_mat,
// 		bool include_mean, const Eigen::VectorXi& seed_chain,
// 		bool display_progress, int nthreads
// 	)
// 	: num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin), nthreads(nthreads),
// 		display_progress(display_progress), sv_objs(num_chains), res(num_chains) {
// 		sv_objs = bvhar::initialize_mcmc<bvhar::McmcSv>(
// 			num_chains, num_iter, x, y,
// 			param_sv, param_prior, param_intercept, param_init, prior_type,
// 			grp_id, own_id, cross_id, grp_mat,
// 			include_mean, seed_chain
// 		);
// 	}
// 	virtual ~SvMcmc() = default;
// 	std::vector<py::dict> returnRecords() {
// 		fit();
// 		return res;
// 	}

// protected:
// 	void runGibbs(int chain) {
// 		bvhar::bvharprogress bar(num_iter, display_progress);
// 		for (int i = 0; i < num_iter; ++i) {
// 			bar.increment();
// 			sv_objs[chain]->doPosteriorDraws();
// 			bar.update();
// 		}
// 	#ifdef _OPENMP
// 		#pragma omp critical
// 	#endif
// 		{
// 			res[chain] = sv_objs[chain]->returnRecords(num_burn, thin);
// 		}
// 	}
// 	void fit() {
// 		if (num_chains == 1) {
// 			runGibbs(0);
// 		} else {
// 		#ifdef _OPENMP
// 			#pragma omp parallel for num_threads(nthreads)
// 		#endif
// 			for (int chain = 0; chain < num_chains; chain++) {
// 				runGibbs(chain);
// 			}
// 		}
// 	}

// private:
// 	int num_chains;
// 	int num_iter;
// 	int num_burn;
// 	int thin;
// 	int nthreads;
// 	bool display_progress;
// 	std::vector<std::unique_ptr<bvhar::McmcSv>> sv_objs;
// 	std::vector<py::dict> res;
// };

PYBIND11_MODULE(_sv, m) {
	// py::class_<SvMcmc>(m, "SvMcmc")
	py::class_<bvhar::McmcRun<bvhar::McmcSv>>(m, "SvMcmc")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&, py::dict&,
			std::vector<py::dict>&, int, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &bvhar::McmcRun<bvhar::McmcSv>::returnRecords);
		// .def("returnRecords", &SvMcmc::returnRecords);
}