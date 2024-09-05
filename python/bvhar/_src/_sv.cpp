#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
// #include <mcmcsv.h>
#include <svforecaster.h>

class SvMcmc {
public:
	SvMcmc(
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
	virtual ~SvMcmc() = default;
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

class SvForecast{
public:
	SvForecast(
		int num_chains, int lag, int step, const Eigen::MatrixXd& y,
		bool sv, bool sparse, py::dict& fit_record,
		Eigen::VectorXi seed_chain, bool include_mean, int nthreads
	)
	: num_chains(num_chains), nthreads(nthreads), sv(sv),
		forecaster(num_chains), density_forecast(num_chains) {
		std::unique_ptr<bvhar::SvRecords> sv_record;
		py::str alpha_name = sparse ? "alpha_sparse_record" : "alpha_record";
		py::str a_name = sparse ? "a_sparse_record" : "a_record";
		for (int i = 0; i < num_chains; ++i) {
			py::list alpha_list = fit_record[alpha_name];
			py::list a_list = fit_record[a_name];
			py::list h_list = fit_record["h_record"];
			py::list sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				py::list c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					py::cast<Eigen::MatrixXd>(alpha_list[i]),
					py::cast<Eigen::MatrixXd>(c_list[i]),
					py::cast<Eigen::MatrixXd>(h_list[i]),
					py::cast<Eigen::MatrixXd>(a_list[i]),
					py::cast<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					py::cast<Eigen::MatrixXd>(alpha_list[i]),
					py::cast<Eigen::MatrixXd>(h_list[i]),
					py::cast<Eigen::MatrixXd>(a_list[i]),
					py::cast<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			forecaster[i].reset(new bvhar::SvVarForecaster(
				*sv_record, step, y, lag, include_mean, static_cast<unsigned int>(seed_chain[i])
			));
		}
	}
	SvForecast(
		int num_chains, int week, int month, int step, const Eigen::MatrixXd& y,
		bool sv, bool sparse, py::dict& fit_record,
		Eigen::VectorXi seed_chain, bool include_mean, int nthreads
	)
	: num_chains(num_chains), nthreads(nthreads), sv(sv),
		forecaster(num_chains), density_forecast(num_chains) {
		std::unique_ptr<bvhar::SvRecords> sv_record;
		py::str alpha_name = sparse ? "alpha_sparse_record" : "alpha_record";
		py::str a_name = sparse ? "a_sparse_record" : "a_record";
		for (int i = 0; i < num_chains; ++i) {
			py::list alpha_list = fit_record[alpha_name];
			py::list a_list = fit_record[a_name];
			py::list h_list = fit_record["h_record"];
			py::list sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				py::list c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					py::cast<Eigen::MatrixXd>(alpha_list[i]),
					py::cast<Eigen::MatrixXd>(c_list[i]),
					py::cast<Eigen::MatrixXd>(h_list[i]),
					py::cast<Eigen::MatrixXd>(a_list[i]),
					py::cast<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					py::cast<Eigen::MatrixXd>(alpha_list[i]),
					py::cast<Eigen::MatrixXd>(h_list[i]),
					py::cast<Eigen::MatrixXd>(a_list[i]),
					py::cast<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			Eigen::MatrixXd har_trans = bvhar::build_vhar(y.cols(), week, month, include_mean);
			forecaster[i].reset(new bvhar::SvVharForecaster(
				*sv_record, step, y, har_trans, month, include_mean, static_cast<unsigned int>(seed_chain[i])
			));
		}
	}
	virtual ~SvForecast() = default;
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
			density_forecast[chain] = forecaster[chain]->forecastDensity(sv);
			forecaster[chain].reset(); // free the memory by making nullptr
		}
	}

private:
	int num_chains;
	int nthreads;
	bool sv;
	std::vector<std::unique_ptr<bvhar::SvForecaster>> forecaster;
	std::vector<Eigen::MatrixXd> density_forecast;
};

PYBIND11_MODULE(_sv, m) {
	py::class_<SvMcmc>(m, "SvMcmc")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&, py::dict&,
			std::vector<py::dict>&, int, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &SvMcmc::returnRecords);
	
	py::class_<SvForecast>(m, "SvForecast")
		.def(py::init<int, int, int, const Eigen::MatrixXd&, bool, bool, py::dict&, const Eigen::VectorXi&, bool, int>())
		.def(py::init<int, int, int, int, const Eigen::MatrixXd&, bool, bool, py::dict&, const Eigen::VectorXi&, bool, int>())
		.def("returnForecast", &SvForecast::returnForecast);
}