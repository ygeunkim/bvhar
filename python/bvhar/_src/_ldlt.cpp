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

class LdltForecast {
public:
	LdltForecast(
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
			py::list a_list = fit_record[a_name];
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
	LdltForecast(
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
			py::list a_list = fit_record[a_name];
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
	virtual ~LdltForecast() = default;
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

class LdltRoll {
public:
	LdltRoll(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, py::dict& fit_record,
		py::dict& param_reg, py::dict& param_prior, py::dict& param_intercept, std::vector<py::dict>& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, int step, const Eigen::MatrixXd& y_test,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, int nthreads, int chunk_size
	)
	: num_window(y.rows()), dim(y.cols()), num_test(y_test.rows()), num_horizon(num_test - step + 1),
		step(step), lag(lag), include_mean(include_mean), sparse(sparse),
		num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin),
		nthreads(nthreads), chunk_size(chunk_size), seed_forecast(seed_forecast),
		roll_mat(num_horizon), roll_y0(num_horizon), y_test(y_test),
		model(num_horizon), forecaster(num_horizon),
		out_forecast(num_horizon, std::vector<Eigen::MatrixXd>(num_chains)),
		lpl_record(Eigen::MatrixXd::Zero(num_horizon, num_chains)) {
		Eigen::MatrixXd tot_mat(num_window + num_test, dim);
		tot_mat << y,
							y_test;
		for (int i = 0; i < num_horizon; ++i) {
			roll_mat[i] = tot_mat.middleRows(i, num_window);
			roll_y0[i] = bvhar::build_y0(roll_mat[i], lag, lag + 1);
		}
		tot_mat.resize(0, 0); // free the memory
		for (auto &reg_chain : model) {
			reg_chain.resize(num_chains);
			for (auto &ptr : reg_chain) {
				ptr = nullptr;
			}
		}
		for (auto &reg_forecast : forecaster) {
			reg_forecast.resize(num_chains);
			for (auto &ptr : reg_forecast) {
				ptr = nullptr;
			}
		}
		std::unique_ptr<bvhar::LdltRecords> record;
		py::str alpha_name = sparse ? "alpha_sparse_record" : "alpha_record";
		py::str a_name = sparse ? "a_sparse_record" : "a_record";
		py::list d_list = fit_record["d_record"];
		py::list alpha_list = fit_record[alpha_name];
		py::list a_list = fit_record[a_name];
		for (int i = 0; i < num_chains; ++i) {
			if (include_mean) {
				py::list c_list = fit_record["c_record"];
				record.reset(new bvhar::LdltRecords(
					py::cast<Eigen::MatrixXd>(alpha_list[i]),
					py::cast<Eigen::MatrixXd>(c_list[i]),
					py::cast<Eigen::MatrixXd>(a_list[i]),
					py::cast<Eigen::MatrixXd>(d_list[i])
				));
			} else {
				record.reset(new bvhar::LdltRecords(
					py::cast<Eigen::MatrixXd>(alpha_list[i]),
					py::cast<Eigen::MatrixXd>(a_list[i]),
					py::cast<Eigen::MatrixXd>(d_list[i])
				));
			}
			forecaster[0][i].reset(new bvhar::RegVarForecaster(
				*record, step, roll_y0[0], lag, include_mean, static_cast<unsigned int>(seed_forecast[i])
			));
		}
		switch (prior_type) {
			case 1: {
				for (int window = 0; window < num_horizon; ++window) {
					Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
					bvhar::MinnParams minn_params(
						num_iter, design, roll_y0[window],
						param_reg, param_prior,
						param_intercept, include_mean
					);
					for (int chain = 0; chain < num_chains; chain++) {
						bvhar::LdltInits sv_inits(param_init[chain]);
						model[window][chain].reset(new bvhar::MinnReg(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
					}
					roll_mat[window].resize(0, 0); // free the memory
				}
				break;
			}
			case 2: {
				for (int window = 0; window < num_horizon; ++window) {
					Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
					bvhar::SsvsParams ssvs_params(
						num_iter, design, roll_y0[window],
						param_reg, grp_id, grp_mat,
						param_prior, param_intercept,
						include_mean
					);
					for (int chain = 0; chain < num_chains; chain++) {
						bvhar::SsvsInits ssvs_inits(param_init[chain]);
						model[window][chain].reset(new bvhar::SsvsReg(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
					}
					roll_mat[window].resize(0, 0); // free the memory
				}
				break;
			}
			case 3: {
				for (int window = 0; window < num_horizon; ++window) {
					Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
					bvhar::HorseshoeParams horseshoe_params(
						num_iter, design, roll_y0[window],
						param_reg, grp_id, grp_mat,
						param_intercept, include_mean
					);
					for (int chain = 0; chain < num_chains; ++chain) {
						bvhar::HsInits hs_inits(param_init[chain]);
						model[window][chain].reset(new bvhar::HorseshoeReg(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
					}
					roll_mat[window].resize(0, 0); // free the memory
				}
				break;
			}
			case 4: {
				for (int window = 0; window < num_horizon; ++window) {
					Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
					bvhar::HierminnParams minn_params(
						num_iter, design, roll_y0[window],
						param_reg,
						own_id, cross_id, grp_mat,
						param_prior,
						param_intercept, include_mean
					);
					for (int chain = 0; chain < num_chains; chain++) {
						bvhar::HierminnInits minn_inits(param_init[chain]);
						model[window][chain].reset(new bvhar::HierminnReg(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
					}
					roll_mat[window].resize(0, 0); // free the memory
				}
				break;
			}
			case 5: {
				for (int window = 0; window < num_horizon; ++window) {
					Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
					bvhar::NgParams ng_params(
						num_iter, design, roll_y0[window],
						param_reg,
						grp_id, grp_mat,
						param_prior, param_intercept,
						include_mean
					);
					for (int chain = 0; chain < num_chains; ++chain) {
						bvhar::NgInits ng_inits(param_init[chain]);
						model[window][chain].reset(new bvhar::NgReg(ng_params, ng_inits, static_cast<unsigned int>(seed_chain(window, chain))));
					}
					roll_mat[window].resize(0, 0); // free the memory
				}
				break;
			}
			case 6: {
				for (int window = 0; window < num_horizon; ++window) {
					Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
					bvhar::DlParams dl_params(
						num_iter, design, roll_y0[window],
						param_reg,
						grp_id, grp_mat,
						param_prior, param_intercept,
						include_mean
					);
					for (int chain = 0; chain < num_chains; ++chain) {
						bvhar::GlInits dl_inits(param_init[chain]);
						model[window][chain].reset(new bvhar::DlReg(dl_params, dl_inits, static_cast<unsigned int>(seed_chain(window, chain))));
					}
					roll_mat[window].resize(0, 0); // free the memory
				}
				break;
			}
		}
		// 
	}
	virtual ~LdltRoll() = default;
	py::dict returnForecast() {
		forecast();
		return py::dict(
			py::arg("forecast") = out_forecast,
			py::arg("lpl") = lpl_record.mean()
		);
	}

protected:
	void runGibbs(int window, int chain) {
		for (int i = 0; i < num_iter; ++i) {
			model[window][chain]->doPosteriorDraws();
		}
		bvhar::LdltRecords record = model[window][chain]->returnLdltRecords(num_burn, thin, sparse);
		forecaster[window][chain].reset(new bvhar::RegVarForecaster(
			record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		));
		model[window][chain].reset(); // free the memory by making nullptr
	}
	void forecast() {
		if (num_chains == 1) {
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads)
		#endif
			for (int window = 0; window < num_horizon; ++window) {
				if (window != 0) {
					runGibbs(window, 0);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				out_forecast[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, 0) = forecaster[window][0]->returnLpl();
				forecaster[window][0].reset(); // free the memory by making nullptr
			}
		} else {
		#ifdef _OPENMP
			#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
		#endif
			for (int window = 0; window < num_horizon; ++window) {
				for (int chain = 0; chain < num_chains; ++chain) {
					if (window != 0) {
						runGibbs(window, 0);
					}
					Eigen::VectorXd valid_vec = y_test.row(step);
					out_forecast[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
					lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
					forecaster[window][chain].reset(); // free the memory by making nullptr
				}
			}
		}
	}

private:
	int num_window;
	int dim;
	int num_test;
	int num_horizon;
	int step;
	int lag;
	bool include_mean;
	bool sparse;
	int num_chains;
	int num_iter;
	int num_burn;
	int thin;
	int nthreads;
	int chunk_size;
	Eigen::VectorXi seed_forecast;
	std::vector<Eigen::MatrixXd> roll_mat;
	std::vector<Eigen::MatrixXd> roll_y0;
	Eigen::MatrixXd y_test;
	std::vector<std::vector<std::unique_ptr<bvhar::McmcReg>>> model;
	std::vector<std::vector<std::unique_ptr<bvhar::RegVarForecaster>>> forecaster;
	std::vector<std::vector<Eigen::MatrixXd>> out_forecast;
	Eigen::MatrixXd lpl_record;
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
	
	py::class_<LdltForecast>(m, "LdltForecast")
		.def(py::init<int, int, int, const Eigen::MatrixXd&, bool, py::dict&, const Eigen::VectorXi&, bool, int>())
		.def(py::init<int, int, int, int, const Eigen::MatrixXd&, bool, py::dict&, const Eigen::VectorXi&, bool, int>())
		.def("returnForecast", &LdltForecast::returnForecast);
	
	py::class_<LdltRoll>(m, "LdltRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, int, const Eigen::MatrixXd&,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, int>()
		)
		.def("returnForecast", &LdltRoll::returnForecast);
}