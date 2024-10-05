#ifndef BAYESFORECAST_H
#define BAYESFORECAST_H

#include "bvharforecaster.h"
#include "bvharomp.h"
// #include "bvharinterrupt.h"

namespace bvhar {

class McmcForecastInterface;
template <typename Forecast, typename Record> class McmcForecast;
class OutForecastInterface;

class McmcForecastInterface {
public:
	virtual ~McmcForecastInterface() = default;
	virtual std::vector<Eigen::MatrixXd> returnForecast() = 0;
};

template <
	typename Forecast = McmcRegForecaster,
	typename Record = typename std::conditional<
		std::is_same<Forecast, McmcRegForecaster>::value,
		LdltRecords2,
		SvRecords2
	>::type
>
class McmcForecast : public McmcForecastInterface {
public:
	McmcForecast(
		int num_chains, int var_lag, int step, Eigen::MatrixXd& response_mat,
		bool sparse, LIST fit_record, int prior_type,
		Eigen::VectorXi& seed_chain, bool include_mean, int nthreads
	)
	: num_chains(num_chains), nthreads(nthreads), forecaster(num_chains), density_forecast(num_chains) {
		std::unique_ptr<RegRecords> reg_record;
		for (int i = 0; i < num_chains; ++i) {
			init_records(reg_record, i, fit_record, include_mean, sparse);
			forecaster[i].reset(new McmcVarForecaster<Forecast, Record>(
				*reg_record, step, response_mat, var_lag, include_mean, static_cast<unsigned int>(seed_chain[i])
			));
		}
	}
	McmcForecast(
		int num_chains, int week, int month, int step, Eigen::MatrixXd response_mat,
		bool sparse, LIST fit_record, int prior_type,
		Eigen::VectorXi seed_chain, bool include_mean, int nthreads
	)
	: num_chains(num_chains), nthreads(nthreads), forecaster(num_chains), density_forecast(num_chains) {
		Eigen::MatrixXd har_trans = build_vhar(response_mat.cols(), week, month, include_mean);
		std::unique_ptr<RegRecords> reg_record;
		for (int i = 0; i < num_chains; ++i) {
			init_records(reg_record, i, fit_record, include_mean, sparse);
			forecaster[i].reset(new McmcVharForecaster<Forecast, Record>(
				*reg_record, step, response_mat, har_trans, month, include_mean, static_cast<unsigned int>(seed_chain[i])
			));
		}
	}
	virtual ~McmcForecast() = default;
	std::vector<Eigen::MatrixXd> returnForecast() override {
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
	std::vector<std::unique_ptr<Forecast>> forecaster;
	std::vector<Eigen::MatrixXd> density_forecast;
};

class OutForecastInterface {
public:
	virtual ~OutForecastInterface() = default;
	virtual void initialize(
		const Eigen::MatrixXd& y, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		const Eigen::MatrixXi& seed_chain
	) = 0;
	virtual LIST returnForecast() = 0;
};

inline void init_mcmcforecaster(
	std::unique_ptr<McmcForecastInterface>& forecaster,
	int num_chains, Eigen::VectorXi& ord, int step, const Eigen::MatrixXd& response_mat,
	bool sparse, double level, LIST& fit_record, int prior_type,
	const Eigen::VectorXi& seed_chain, bool include_mean, int nthreads
) {
	if (CONTAINS(fit_record, "sigh_record")) {
		if (ord.size() == 1) {
			forecaster.reset(new McmcForecast<McmcSvForecaster>(
				num_chains, ord[0], step, response_mat, sparse, fit_record, prior_type,
				seed_chain, include_mean, nthreads
			));
		} else if (ord.size() == 2) {
			forecaster.reset(new McmcForecast<McmcSvForecaster>(
				num_chains, ord[0], ord[1], step, response_mat, sparse, fit_record, prior_type,
				seed_chain, include_mean, nthreads
			));
		} else {
			STOP("'ord' should be length 1 when VAR, or length 2 when VHAR");
		}
	} else {
		if (ord.size() == 1) {
			forecaster.reset(new McmcForecast<McmcRegForecaster>(
				num_chains, ord[0], step, response_mat, sparse, fit_record, prior_type,
				seed_chain, include_mean, nthreads
			));
		} else if (ord.size() == 2) {
			forecaster.reset(new McmcForecast<McmcRegForecaster>(
				num_chains, ord[0], ord[1], step, response_mat, sparse, fit_record, prior_type,
				seed_chain, include_mean, nthreads
			));
		} else {
			STOP("'ord' should be length 1 when VAR, or length 2 when VHAR");
		}
	}
}

}; // namespace bvhar

#endif // BAYESFORECAST_H