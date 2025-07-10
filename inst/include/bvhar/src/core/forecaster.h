#ifndef BVHAR_CORE_FORECASTER_H
#define BVHAR_CORE_FORECASTER_H

#include "./common.h"
#include "./omp.h"

namespace bvhar {

template <typename ReturnType, typename DataType> class MultistepForecaster;
template <typename ReturnType, typename DataType> class MultistepForecastRun;
template <typename ReturnType, typename DataType> class ExogenForecaster;
template <typename ReturnType, typename DataType> class AutoregGenerator;

/**
 * @brief Base class for Recursive multi-step forecasting
 * 
 * @tparam ReturnType Type of forecasting result.
 * @tparam DataType Type of one unit.
 */
template <typename ReturnType = Eigen::MatrixXd, typename DataType = Eigen::VectorXd>
class MultistepForecaster {
public:
	MultistepForecaster(int step, const ReturnType& response, int lag)
	: step(step), lag(lag), response(response), debug_logger(BVHAR_DEBUG_LOGGER("MultistepForecaster")) {
		BVHAR_INIT_DEBUG(debug_logger);
    BVHAR_DEBUG_LOG(debug_logger, "Constructor: step={}, lag={}", step, lag);
	}
	virtual ~MultistepForecaster() {
		BVHAR_DEBUG_DROP("MultistepForecaster");
	}

	/**
	 * @brief Return the forecast result.
	 * Bayes methods return density forecast while frequentist methods return point forecast.
	 * 
	 * @return ReturnType 
	 */
	ReturnType doForecast() {
		BVHAR_DEBUG_LOG(debug_logger, "doForecast() called");
		forecast();
		return pred_save;
	}

	ReturnType doForecast(const DataType& valid_vec) {
		BVHAR_DEBUG_LOG(debug_logger, "doForecast(valid_vec) called");
		forecast(valid_vec);
		return pred_save;
	}

	virtual DataType getLastForecast() = 0;
	virtual DataType getLastForecast(const DataType& valid_vec) { return getLastForecast(); }

protected:
	int step, lag;
	ReturnType response;
	ReturnType pred_save; // when Point: rbind(step) or when Density: rbind(step), cbind(sims)
	DataType point_forecast; // y_(T + h - 1)
	DataType last_pvec; // [ y_(T + h - 1)^T, y_(T + h - 2)^T, ..., y_(T + h - p)^T, 1 ] (1 when constant term)
	DataType tmp_vec; // y_(T + h - 2), ... y_(T + h - lag)
	std::shared_ptr<spdlog::logger> debug_logger;

	/**
	 * @brief Initialize lagged predictors 'point_forecast', 'last_pvec' and 'tmp_vec'.
	 * 
	 */
	virtual void initLagged() = 0;

	/**
	 * @brief Multi-step forecasting
	 * 
	 */
	virtual void forecast() {
		BVHAR_DEBUG_LOG(debug_logger, "forecast() called");
		for (int h = 0; h < step; ++h) {
			BVHAR_DEBUG_LOG(debug_logger, "h={} / step={}", h, step);
			setRecursion();
			updatePred(h, 0);
			updateRecursion();
		}
	}

	virtual void forecast(const DataType& valid_vec) { forecast(); }

	/**
	 * @brief Set the initial lagged unit
	 * 
	 */
	virtual void setRecursion() = 0;

	/**
	 * @brief Move the lagged unit element based on one-step ahead forecasting
	 * 
	 */
	virtual void updateRecursion() = 0;
	// virtual void forecastOut(const int i) = 0;

	/**
	 * @brief Compute Linear predictor
	 * 
	 */
	virtual void updatePred(const int h, const int i) = 0;
};

/**
 * @brief Base class for multi-step forecasting runner
 * 
 * @tparam ReturnType 
 * @tparam DataType 
 */
template <typename ReturnType = Eigen::MatrixXd, typename DataType = Eigen::VectorXd>
class MultistepForecastRun {
public:
	MultistepForecastRun() : debug_logger(BVHAR_DEBUG_LOGGER("MultistepForecastRun")) {
		BVHAR_INIT_DEBUG(debug_logger);
    BVHAR_DEBUG_LOG(debug_logger, "Constructor");
	}
	virtual ~MultistepForecastRun() = default;

	/**
	 * @brief Forecast
	 * 
	 */
	virtual void forecast() {}

protected:
	std::shared_ptr<spdlog::logger> debug_logger;
};

/**
 * @brief Base class for exogenous variables in forecaster
 * 
 * @tparam ReturnType 
 * @tparam DataType 
 */
template <typename ReturnType = Eigen::MatrixXd, typename DataType = Eigen::VectorXd>
class ExogenForecaster {
public:
	ExogenForecaster() : debug_logger(BVHAR_DEBUG_LOGGER("ExogenForecaster")) {
		BVHAR_INIT_DEBUG(debug_logger);
    BVHAR_DEBUG_LOG(debug_logger, "Default Constructor");
	}
	ExogenForecaster(int lag, const ReturnType& exogen)
	: lag(lag), exogen(exogen), debug_logger(BVHAR_DEBUG_LOGGER("ExogenForecaster")) {
		BVHAR_INIT_DEBUG(debug_logger);
    BVHAR_DEBUG_LOG(debug_logger, "Constructor: lag={}", lag);
	}
	virtual ~ExogenForecaster() = default;

	/**
	 * @brief Add point forecast by exogenous terms
	 * 
	 * @param point_forecast 
	 * @param h 
	 */
	virtual void appendForecast(DataType& point_forecast, const int h) {}

protected:
	int lag;
	ReturnType exogen;
	DataType last_pvec;
	std::shared_ptr<spdlog::logger> debug_logger;
};

/**
 * @brief Base class for DGP feature in forecaster classes
 * 
 * @tparam ReturnType 
 * @tparam DataType 
 */
template <typename ReturnType = Eigen::MatrixXd, typename DataType = Eigen::VectorXd>
class AutoregGenerator {
public:
	AutoregGenerator() : debug_logger(BVHAR_DEBUG_LOGGER("AutoregGenerator")) {
		BVHAR_INIT_DEBUG(debug_logger);
    BVHAR_DEBUG_LOG(debug_logger, "Default Constructor");
	}
	AutoregGenerator(int num_iter, unsigned int seed)
	: num_iter(num_iter), rng(seed), debug_logger(BVHAR_DEBUG_LOGGER("ExogenForecaster")) {
		BVHAR_INIT_DEBUG(debug_logger);
    BVHAR_DEBUG_LOG(debug_logger, "Constructor: num_iter={}", num_iter);
	}
	virtual ~AutoregGenerator() = default;

	/**
	 * @brief Add error term to point forecast
	 * 
	 * @param point_forecast 
	 * @param h 
	 */
	virtual void appendError(DataType& point_forecast) {}

protected:
	int num_iter;
	BHRNG rng;
	// int lag;
	// ReturnType exogen;
	// DataType last_pvec;
	DataType error_term; // --> Change type
	std::shared_ptr<spdlog::logger> debug_logger;
};

} // namespace bvhar

#endif // BVHAR_CORE_FORECASTER_H