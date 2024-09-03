#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
// #include <mcmcreg.h>
#include <regforecaster.h>

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
	: num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin), nthreads(nthreads),
		display_progress(display_progress), sur_objs(num_chains), res(num_chains) {
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
		for (int i = 0; i < num_iter; ++i) {
			bar.increment();
			sur_objs[chain]->doPosteriorDraws();
			bar.update();
		}
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
	int num_chains;
	int num_iter;
	int num_burn;
	int thin;
	int nthreads;
	bool display_progress;
	std::vector<std::unique_ptr<bvhar::McmcReg>> sur_objs;
	std::vector<py::dict> res;
};

class LdltForecaster {
public:
	LdltForecaster(
		int num_chains, int lag, int step, const Eigen::MatrixXd& y,
		bool sparse, py::dict& fit_record,
		const Eigen::VectorXi& seed_chain, bool include_mean, int nthreads
	)
	: num_chains(num_chains), nthreads(nthreads),
		forecaster(num_chains), density_forecast(num_chains) {
		std::unique_ptr<bvhar::LdltRecords> reg_record;
		py::str alpha_name = sparse ? "alpha_sparse_record" : "alpha_record";
		py::str a_name = sparse ? "a_sparse_record" : "a_record";
		for (int i = 0; i < num_chains; ++i) {
			py::list alpha_list = fit_record[alpha_name];
			// py::list alpha_list = fit_record[alpha_name].cast<py::list>();
			py::list a_list = fit_record["a_record"];
			py::list d_list = fit_record["d_record"];
			if (include_mean) {
				py::list c_list = fit_record["c_record"];
				reg_record.reset(new bvhar::LdltRecords(
					py::cast<Eigen::MatrixXd>(alpha_list[i]),
					py::cast<Eigen::MatrixXd>(c_list[i]),
					py::cast<Eigen::MatrixXd>(a_list[i]),
					py::cast<Eigen::MatrixXd>(d_list[i])
				));
			} else {
				reg_record.reset(new bvhar::LdltRecords(
					py::cast<Eigen::MatrixXd>(alpha_list[i]),
					py::cast<Eigen::MatrixXd>(a_list[i]),
					py::cast<Eigen::MatrixXd>(d_list[i])
				));
			}
			forecaster[i].reset(new bvhar::RegVarForecaster(
				*reg_record, step, y, lag, include_mean, static_cast<unsigned int>(seed_chain[i])
			));
		}
	}
	LdltForecaster(
		int num_chains, int week, int month, int step, const Eigen::MatrixXd& y,
		bool sparse, py::dict& fit_record,
		const Eigen::VectorXi& seed_chain, bool include_mean, int nthreads
	)
	: num_chains(num_chains), nthreads(nthreads),
		forecaster(num_chains), density_forecast(num_chains) {
		std::unique_ptr<bvhar::LdltRecords> reg_record;
		py::str alpha_name = sparse ? "alpha_sparse_record" : "alpha_record";
		py::str a_name = sparse ? "a_sparse_record" : "a_record";
		for (int i = 0; i < num_chains; ++i) {
			py::list alpha_list = fit_record[alpha_name];
			// py::list alpha_list = fit_record[alpha_name].cast<py::list>();
			py::list a_list = fit_record["a_record"];
			py::list d_list = fit_record["d_record"];
			if (include_mean) {
				py::list c_list = fit_record["c_record"];
				reg_record.reset(new bvhar::LdltRecords(
					py::cast<Eigen::MatrixXd>(alpha_list[i]),
					py::cast<Eigen::MatrixXd>(c_list[i]),
					py::cast<Eigen::MatrixXd>(a_list[i]),
					py::cast<Eigen::MatrixXd>(d_list[i])
				));
			} else {
				reg_record.reset(new bvhar::LdltRecords(
					py::cast<Eigen::MatrixXd>(alpha_list[i]),
					py::cast<Eigen::MatrixXd>(a_list[i]),
					py::cast<Eigen::MatrixXd>(d_list[i])
				));
			}
			Eigen::MatrixXd har_trans = bvhar::build_vhar(y.cols(), week, month, include_mean);
			forecaster[i].reset(new bvhar::RegVharForecaster(
				*reg_record, step, y, har_trans, month, include_mean, static_cast<unsigned int>(seed_chain[i])
			));
		}
	}
	virtual ~LdltForecaster() = default;
	std::vector<Eigen::MatrixXd> returnForecast() {
		forecast();
		return density_forecast;
	}

protected:
	void forecast() {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; ++chain) {
			density_forecast[chain] = forecaster[chain]->forecastDensity();
			forecaster[chain].reset(); // free the memory by making nullptr
		}
	}

private:
	int num_chains;
	int nthreads;
	std::vector<std::unique_ptr<bvhar::RegForecaster>> forecaster;
	std::vector<Eigen::MatrixXd> density_forecast;
};

PYBIND11_MODULE(_ldlt, m) {
	py::class_<McmcLdlt>(m, "McmcLdlt")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&, py::dict&,
			std::vector<py::dict>&, int, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &McmcLdlt::returnRecords);
	
	py::class_<LdltForecaster>(m, "LdltForecaster")
		.def(py::init<int, int, int, const Eigen::MatrixXd&, bool, py::dict&, const Eigen::VectorXi&, bool, int>())
		.def(py::init<int, int, int, int, const Eigen::MatrixXd&, bool, py::dict&, const Eigen::VectorXi&, bool, int>())
		.def("returnForecast", &LdltForecaster::returnForecast);
}