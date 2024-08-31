#include <pybind11/eigen.h>
#include <mcmcreg.h>

class McmcLdlt {
public:
	McmcLdlt(
		int num_chains, int num_iter, int num_burn, int thin,
		const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		py::dict& param_reg, py::dict& param_prior, py::dict& param_intercept,
		std::vector<py::dict>& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::MatrixXi& grp_mat,
		bool include_mean, const Eigen::VectorXi& seed_chain,
		bool display_progress, int nthreads
	)
	: prior(prior_type),
		num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin),
		display_progress(display_progress),
		sur_objs(num_chains, nullptr), res(num_chains) {
		switch (prior_type) {
			case 1: {
				bvhar::MinnParams minn_params(
					num_iter, x, y,
					param_reg, param_prior,
					param_intercept, include_mean
				);
				for (int i = 0; i < num_chains; i++ ) {
					bvhar::LdltInits ldlt_inits(param_init[i]);
					sur_objs[i].reset(new bvhar::MinnReg(minn_params, ldlt_inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 2: {
				bvhar::SsvsParams ssvs_params(
					num_iter, x, y,
					param_reg,
					grp_id, grp_mat,
					param_prior,
					param_intercept,
					include_mean
				);
				for (int i = 0; i < num_chains; i++ ) {
					bvhar::SsvsInits ssvs_inits(param_init[i]);
					sur_objs[i].reset(new bvhar::SsvsReg(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 3: {
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, x, y,
					param_reg,
					grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int i = 0; i < num_chains; i++ ) {
					bvhar::HsInits hs_inits(param_init[i]);
					sur_objs[i].reset(new bvhar::HorseshoeReg(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 4: {
				bvhar::HierminnParams minn_params(
					num_iter, x, y,
					param_reg,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int i = 0; i < num_chains; i++ ) {
					bvhar::HierminnInits minn_inits(param_init[i]);
					sur_objs[i].reset(new bvhar::HierminnReg(minn_params, minn_inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 5: {
				bvhar::NgParams ng_params(
					num_iter, x, y,
					param_reg,
					grp_id, grp_mat,
					param_prior,
					param_intercept,
					include_mean
				);
				for (int i = 0; i < num_chains; ++i) {
					bvhar::NgInits ng_inits(param_init[i]);
					sur_objs[i].reset(new bvhar::NgReg(ng_params, ng_inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
			case 6: {
				bvhar::DlParams dl_params(
					num_iter, x, y,
					param_reg,
					grp_id, grp_mat,
					param_prior,
					param_intercept,
					include_mean
				);
				for (int i = 0; i < num_chains; ++i) {
					bvhar::GlInits dl_inits(param_init[i]);
					sur_objs[i].reset(new bvhar::DlReg(dl_params, dl_inits, static_cast<unsigned int>(seed_chain[i])));
				}
				break;
			}
		}
	}
	virtual ~McmcLdlt() = default;
	std::vector<py::dict> returnRecords() {
		fit();
		return res;
	}

protected:
	void runGibbs(int chain) {
		bvhar::bvharprogress bar(num_iter, display_progress);
		bar.increment();
		sur_objs[chain]->doPosteriorDraws();
		bar.update();
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = sur_objs[chain]->returnRecords(num_burn, thin);
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
	int prior;
	int num_chains;
	int num_iter;
	int num_burn;
	int thin;
	bool display_progress;
	std::vector<std::unique_ptr<bvhar::McmcReg>> sur_objs;
	std::vector<py::dict> res;
};


PYBIND11_MODULE(_ldlt, m) {
	m.doc() = "MCMC for VAR-LDLT and VHAR-LDLT";

	// py::class_<bvhar::RegParams>(m, "RegParams")
	// 	.def(py::init<int, const Eigen::MatrixXd&, const Eigen::MatrixXd&, py::dict&, py::dict&, bool>())
	// 	.def_readwrite("_iter", &bvhar::RegParams::_iter)
	// 	.def_readwrite("_x", &bvhar::RegParams::_x)
	// 	.def_readwrite("_y", &bvhar::RegParams::_y)
	// 	.def_readwrite("_sig_shp", &bvhar::RegParams::_sig_shp)
	// 	.def_readwrite("_sig_scl", &bvhar::RegParams::_sig_scl)
	// 	.def_readwrite("_sd_non", &bvhar::RegParams::_sd_non)
	// 	.def_readwrite("_mean", &bvhar::RegParams::_mean);
	
	// py::class_<bvhar::RegInits>(m, "RegInits")
	// 	.def(py::init<py::dict&>())
	// 	.def_readwrite("_coef", &bvhar::RegInits::_coef)
	// 	.def_readwrite("_contem", &bvhar::RegInits::_contem);

	// py::class_<bvhar::LdltInits, bvhar::RegInits>(m, "LdldInits")
	// 	.def(py::init<py::dict&>())
	// 	.def_readwrite("_diag", &bvhar::LdltInits::_diag);

  // py::class_<bvhar::McmcReg>(m, "McmcReg")
  //   .def(py::init<const bvhar::RegParams&, const bvhar::LdltInits&, unsigned int>())
	// 	.def("returnRecords", &bvhar::McmcReg::returnRecords);

	py::class_<McmcLdlt>(m, "McmcLdlt")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&, py::dict&,
			std::vector<py::dict>&, int, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("fit", &McmcLdlt::fit)
		.def("returnRecords", &McmcLdlt::returnRecords);
}