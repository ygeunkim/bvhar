#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <svforecaster.h>

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

PYBIND11_MODULE(_svforecast, m) {
	py::class_<SvForecast>(m, "SvForecast")
		.def(py::init<int, int, int, const Eigen::MatrixXd&, bool, bool, py::dict&, const Eigen::VectorXi&, bool, int>())
		.def(py::init<int, int, int, int, const Eigen::MatrixXd&, bool, bool, py::dict&, const Eigen::VectorXi&, bool, int>())
		.def("returnForecast", &SvForecast::returnForecast);
}