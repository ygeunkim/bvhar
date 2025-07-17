#ifndef BVHAR_BAYES_FORECASTER_H
#define BVHAR_BAYES_FORECASTER_H

#include "../core/forecaster.h"
#include "./bayes.h"

namespace bvhar {

template <typename ReturnType, typename DataType> class BayesForecaster;
template <typename ReturnType, typename DataType> class McmcForecastRun;
class McmcOutforecastInterface;
template <typename ReturnType, typename DataType, bool isUpdate> class McmcOutforecastRun;

/**
 * @brief Base class for forecaster of Bayesian methods
 * 
 * @tparam ReturnType 
 * @tparam DataType 
 */
template <typename ReturnType = Eigen::MatrixXd, typename DataType = Eigen::VectorXd>
class BayesForecaster : public MultistepForecaster<ReturnType, DataType> {
public:
	BayesForecaster(int step, const ReturnType& response, int lag, int num_sim, unsigned int seed)
	: MultistepForecaster<ReturnType, DataType>(step, response, lag),
		lpl(Eigen::VectorXd::Zero(step)), num_sim(num_sim), rng(seed) {
    BVHAR_DEBUG_LOG(debug_logger, "BayesForecaster Constructor: step={}, lag={}, num_sim={}", step, lag, num_sim);
	}
	virtual ~BayesForecaster() = default;
	// using MultistepForecaster<ReturnType, DataType>::returnForecast();
	
	/**
	 * @brief Return the draws of LPL
	 * 
	 * @return Eigen::VectorXd LPL draws
	 */
	Eigen::VectorXd returnLplRecord() {
		return lpl;
	}

	/**
	 * @brief Return ALPL
	 * 
	 * @return double ALPL value
	 */
	double returnLpl() {
		return lpl.mean();
	}

protected:
	using MultistepForecaster<ReturnType, DataType>::step;
	using MultistepForecaster<ReturnType, DataType>::lag;
	using MultistepForecaster<ReturnType, DataType>::response;
	using MultistepForecaster<ReturnType, DataType>::pred_save; // rbind(step), cbind(sims)
	using MultistepForecaster<ReturnType, DataType>::point_forecast;
	using MultistepForecaster<ReturnType, DataType>::last_pvec;
	using MultistepForecaster<ReturnType, DataType>::tmp_vec;
	using MultistepForecaster<ReturnType, DataType>::debug_logger;
	Eigen::VectorXd lpl;
	std::mutex mtx;
	int num_sim;
	BHRNG rng;

	void forecast() override {
		std::lock_guard<std::mutex> lock(mtx);
		BVHAR_DEBUG_LOG(debug_logger, "forecast() called");
		DataType obs_vec = last_pvec; // y_T, y_(T - 1), ... y_(T - lag + 1)
		for (int i = 0; i < num_sim; ++i) {
			BVHAR_DEBUG_LOG(debug_logger, "i={} / num_sim={}", i, num_sim);
			initRecursion(obs_vec);
			updateParams(i);
			forecastOut(i);
		}
	}

	void forecast(const DataType& valid_vec) override {
		std::lock_guard<std::mutex> lock(mtx);
		BVHAR_DEBUG_LOG(debug_logger, "forecast(valid_vec) called");
		DataType obs_vec = last_pvec; // y_T, y_(T - 1), ... y_(T - lag + 1)
		for (int i = 0; i < num_sim; ++i) {
			BVHAR_DEBUG_LOG(debug_logger, "i={} / num_sim={}", i, num_sim);
			initRecursion(obs_vec);
			updateParams(i);
			forecastOut(i, valid_vec);
		}
		lpl.array() /= num_sim;
	}

	/**
	 * @brief Initialize lagged predictor for each MCMC loop.
	 * 
	 * @param obs_vec 
	 */
	virtual void initRecursion(const DataType& obs_vec) = 0;

	/**
	 * @brief Update members with corresponding MCMC draw
	 * 
	 * @param i MCMC step
	 */
	virtual void updateParams(const int i) = 0;

	/**
	 * @brief Draw i-th forecast
	 * 
	 * @param i MCMC step
	 */
	void forecastOut(const int i) {
		for (int h = 0; h < step; ++h) {
			this->setRecursion();
			this->updatePred(h, i);
			this->updateRecursion();
		}
	}

	/**
	 * @brief Compute LPL
	 * 
	 * @param h Forecast step
	 * @param valid_vec Validation vector
	 */
	virtual void updateLpl(int h, const DataType& valid_vec) = 0;

	void forecastOut(const int i, const DataType& valid_vec) {
		for (int h = 0; h < step; ++h) {
			this->setRecursion();
			this->updatePred(h, i);
			updateLpl(h, valid_vec);
			this->updateRecursion();
		}
	}
};

/**
 * @brief Base runner class for MCMC forecaster
 * 
 * @tparam ReturnType 
 * @tparam DataType 
 */
template <typename ReturnType = Eigen::MatrixXd, typename DataType = Eigen::VectorXd>
class McmcForecastRun : public MultistepForecastRun<ReturnType, DataType> {
public:
	McmcForecastRun(int num_chains, int lag, int step, int nthreads)
	: num_chains(num_chains), nthreads(nthreads),
		density_forecast(num_chains), forecaster(num_chains) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"McmcForecastRun Constructor: num_chains={}, lag={}, step={}, nthreads={}",
			num_chains, lag, step, nthreads
		);
	}
	virtual ~McmcForecastRun() = default;

	/**
	 * @brief Forecast
	 * 
	 */
	void forecast() override {
		BVHAR_DEBUG_LOG(debug_logger, "forecast() called");
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; ++chain) {
			BVHAR_DEBUG_LOG(debug_logger, "[Thread {}] chain={} / num_chains={}", std::to_string(omp_get_thread_num()), chain, num_chains);
			density_forecast[chain] = forecaster[chain]->doForecast();
			forecaster[chain].reset();
		}
	}

	/**
	 * @brief Return forecast draws
	 * 
	 * @return std::vector<ReturnType> Forecast density of each chain
	 */
	std::vector<ReturnType> returnForecast() {
		forecast();
		return density_forecast;
	}

private:
	int num_chains, nthreads;
	std::vector<ReturnType> density_forecast;

protected:
	std::vector<std::unique_ptr<BayesForecaster<ReturnType, DataType>>> forecaster;
	using MultistepForecastRun<ReturnType, DataType>::debug_logger;
};

class McmcOutforecastInterface {
public:
	McmcOutforecastInterface() {}
	virtual ~McmcOutforecastInterface() = default;
	
	/**
	 * @brief Out-of-sample forecasting
	 * 
	 */
	virtual void forecast() = 0;

	/**
	 * @brief Return out-of-sample forecasting draws
	 * 
	 * @return BVHAR_LIST `BVHAR_LIST` containing forecast draws. Include ALPL when `get_lpl` is `true`.
	 */
	virtual BVHAR_LIST returnForecast() = 0;
};

/**
 * @brief Base class for pseudo out-of-sample forecasting
 * 
 * @tparam ReturnType 
 * @tparam DataType 
 */
template <typename ReturnType = Eigen::MatrixXd, typename DataType = Eigen::VectorXd, bool isUpdate = true>
class McmcOutForecastRun : public McmcOutforecastInterface {
public:
	McmcOutForecastRun(
		int num_window, int lag,
		int num_chains, int num_iter, int num_burn, int thin,
		int step, const ReturnType& y_test, bool get_lpl,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
		Optional<int> exogen_lag = NULLOPT
	)
	: num_window(num_window), num_test(y_test.rows()), num_horizon(num_test - step + 1), step(step),
		lag(lag), num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin), nthreads(nthreads),
		get_lpl(get_lpl), display_progress(display_progress),
		seed_forecast(seed_forecast), roll_mat(num_horizon), roll_y0(num_horizon), y_test(y_test),
		model(num_horizon), forecaster(num_horizon),
		// out_forecast(num_horizon, std::vector<ReturnType>(num_chains)),
		out_forecast(num_horizon, std::vector<DataType>(num_chains)),
		lpl_record(Eigen::MatrixXd::Zero(num_horizon, num_chains)),
		roll_exogen_mat(num_horizon), roll_exogen(num_horizon), lag_exogen(exogen_lag),
		debug_logger(BVHAR_DEBUG_LOGGER("McmcOutForecastRun")) {
		BVHAR_INIT_DEBUG(debug_logger);
    BVHAR_DEBUG_LOG(
			debug_logger, "Constructor: num_window={}, lag={}, step={}, num_test={}, num_iter={}, num_burn={}, thin={}, nthreads={}",
			num_window, lag, step, num_test, num_iter, num_burn, thin, nthreads
		);
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
		for (int i = 0; i < num_horizon; ++i) {
			roll_exogen_mat[i] = NULLOPT;
			roll_exogen[i] = NULLOPT;
		}
	}
	virtual ~McmcOutForecastRun() = default;

	/**
	 * @brief Out-of-sample forecasting
	 * 
	 */
	void forecast() override {
		BVHAR_DEBUG_LOG(debug_logger, "forecast() called");
		if (num_chains == 1) {
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads)
		#endif
			for (int window = 0; window < num_horizon; ++window) {
				forecastWindow(window, 0);
			}
		} else {
		#ifdef _OPENMP
			#pragma omp parallel for collapse(2) schedule(static, num_chains) num_threads(nthreads)
		#endif
			for (int window = 0; window < num_horizon; ++window) {
				for (int chain = 0; chain < num_chains; ++chain) {
					forecastWindow(window, chain);
				}
			}
		}
	}

	/**
	 * @brief Return out-of-sample forecasting draws
	 * 
	 * @return BVHAR_LIST `BVHAR_LIST` containing forecast draws. Include ALPL when `get_lpl` is `true`.
	 */
	BVHAR_LIST returnForecast() override {
		forecast();
		BVHAR_LIST res = BVHAR_CREATE_LIST(BVHAR_NAMED("forecast") = BVHAR_WRAP(out_forecast));
		if (get_lpl) {
			res["lpl"] = lpl_record;
		}
		return res;
	}
	
protected:
	int num_window, num_test, num_horizon, step;
	int lag, num_chains, num_iter, num_burn, thin, nthreads;
	bool get_lpl, display_progress;
	Eigen::VectorXi seed_forecast;
	std::vector<ReturnType> roll_mat;
	std::vector<ReturnType> roll_y0;
	ReturnType y_test;
	std::vector<std::vector<std::unique_ptr<McmcAlgo>>> model;
	std::vector<std::vector<std::unique_ptr<BayesForecaster<ReturnType, DataType>>>> forecaster;
	// std::vector<std::vector<ReturnType>> out_forecast;
	std::vector<std::vector<DataType>> out_forecast;
	Eigen::MatrixXd lpl_record;
	std::vector<Optional<ReturnType>> roll_exogen_mat;
	std::vector<Optional<ReturnType>> roll_exogen;
	Optional<int> lag_exogen;
	std::shared_ptr<spdlog::logger> debug_logger;

	/**
	 * @brief Replace the forecast smart pointer given MCMC result
	 * 
	 * @param model MCMC model
	 * @param window Window index
	 * @param chain Chain index
	 */
	virtual void updateForecaster(int window, int chain) = 0;

	/**
	 * @brief Get valid_vec
	 * 
	 * @return DataType 
	 */
	virtual DataType getValid() = 0;

	/**
	 * @brief Conduct MCMC and update forecast pointer
	 * 
	 * @param window Window index
	 * @param chain Chain index
	 */
	void runGibbs(int window, int chain) {
		BVHAR_DEBUG_LOG(debug_logger, "runGibbs(window={}, chain={}) called", window, chain);
		std::string log_name = fmt::format("Chain {} / Window {}", chain + 1, window + 1);
		auto logger = spdlog::get(log_name);
		if (logger == nullptr) {
			logger = BVHAR_SPDLOG_SINK_MT(log_name);
		}
		logger->set_pattern("[%n] [Thread " + std::to_string(omp_get_thread_num()) + "] %v");
		int logging_freq = num_iter / 20; // 5 percent
		if (logging_freq == 0) {
			logging_freq = 1;
		}
		BVHAR_INIT_DEBUG(logger);
		bvharinterrupt();
		for (int i = 0; i < num_burn; ++i) {
			model[window][chain]->doWarmUp();
			BVHAR_DEBUG_LOG(logger, "{} / {} (Warmup)", i + 1, num_iter);
			if (display_progress && (i + 1) % logging_freq == 0) {
				logger->info("{} / {} (Warmup)", i + 1, num_iter);
			}
		}
		logger->flush();
		for (int i = num_burn; i < num_iter; ++i) {
			if (bvharinterrupt::is_interrupted()) {
				logger->warn("User interrupt in {} / {}", i + 1, num_iter);
				break;
			}
			model[window][chain]->doPosteriorDraws();
			BVHAR_DEBUG_LOG(logger, "{} / {} (Sampling)", i + 1, num_iter);
			if (display_progress && (i + 1) % logging_freq == 0) {
				logger->info("{} / {} (Sampling)", i + 1, num_iter);
			}
		}
		// RecordType reg_record = model[window][chain]->template returnStructRecords<RecordType>(0, thin, sparse);
		// updateForecaster(reg_record, window, chain);
		// model[window][chain].reset();
		updateForecaster(window, chain);
		logger->flush();
		spdlog::drop(log_name);
	}

	/**
	 * @brief Forecast
	 * 
	 * @param window Window index
	 * @param chain Chain index
	 */
	void forecastWindow(int window, int chain) {
		BVHAR_DEBUG_LOG(debug_logger, "forecastWindow(window={}, chain={}) called", window, chain);
		using is_mcmc = std::integral_constant<bool, isUpdate>;
		if (window != 0 && is_mcmc::value) {
			runGibbs(window, chain);
		}
		// Eigen::VectorXd valid_vec = y_test.row(step);
		DataType valid_vec = getValid();
		// out_forecast[window][chain] = forecaster[window][chain]->doForecast(valid_vec).bottomRows(1);
		out_forecast[window][chain] = forecaster[window][chain]->getLastForecast(valid_vec);
		lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
		forecaster[window][chain].reset(); // free the memory by making nullptr
	}
};

} // namespace bvhar

#endif // BVHAR_BAYES_FORECASTER_H