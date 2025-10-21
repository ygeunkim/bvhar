#ifndef BVHAR_BAYES_TRIANGULAR_FORECASTER_H
#define BVHAR_BAYES_TRIANGULAR_FORECASTER_H

#include "./triangular.h"
#include "../forecaster.h"

namespace bvhar {

class CtaExogenForecaster;
class CtaForecaster;
class RegForecaster;
class SvForecaster;
template <typename BaseForecaster> class CtaVarForecaster;
template <typename BaseForecaster> class CtaVharForecaster;
template <typename BaseForecaster> class CtaVarSelectForecaster;
template <typename BaseForecaster> class CtaVharSelectForecaster;
// Running forecasters
template <typename BaseForecaster> class CtaForecastRun;
template <typename BaseForecaster> class CtaOutforecastInterface;
template <typename BaseForecaster, bool> class CtaOutforecastRun;
template <typename BaseForecaster, bool, bool> class CtaRollforecastRun;
template <typename BaseForecaster, bool, bool> class CtaExpandforecastRun;
template <template <typename, bool, bool> class BaseOutForecast, typename BaseForecaster, bool, bool> class CtaVarforecastRun;
template <template <typename, bool, bool> class BaseOutForecast, typename BaseForecaster, bool, bool> class CtaVharforecastRun;

class CtaExogenForecaster : public ExogenForecaster<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	CtaExogenForecaster() {}
	CtaExogenForecaster(int lag, const Eigen::MatrixXd& exogen, int dim)
	: ExogenForecaster<Eigen::MatrixXd, Eigen::VectorXd>(lag, exogen),
		dim(dim), dim_exogen(exogen.cols()), nrow_exogen(dim_exogen * (lag + 1)), num_exogen(dim * nrow_exogen),
		coef_mat(nrow_exogen, dim) {
    BVHAR_DEBUG_LOG(debug_logger, "Constructor: dim={}", dim);
		last_pvec = vectorize_eigen(exogen.topRows(lag + 1).colwise().reverse().transpose().eval()); // x_(T + h), ..., x_(T + h - s)
	}
	virtual ~CtaExogenForecaster() = default;

	void appendForecast(Eigen::VectorXd& point_forecast, const int h) override {
		BVHAR_DEBUG_LOG(debug_logger, "appendForecast(point_forecast, h) called");
		last_pvec = vectorize_eigen(exogen.middleRows(h, lag + 1).colwise().reverse().transpose().eval()); // x_(T + h), ..., x_(T + h - s)
		point_forecast += coef_mat.transpose() * last_pvec;
	}

	void appendPredict(Eigen::MatrixXd& pred, const Eigen::MatrixXd& design) {
		pred += design.rightCols(nrow_exogen) * coef_mat;
	}

	Eigen::VectorXd getObs() {
		BVHAR_DEBUG_LOG(debug_logger, "getObs() called");
		return last_pvec;
	}

	int getSize() {
		return num_exogen;
	}

	void updateCoefmat(const Eigen::VectorXd& coef_record) {
		BVHAR_DEBUG_LOG(debug_logger, "updateCoefmat() called: num_exogen={}, dim={}", num_exogen, dim);
		coef_mat = unvectorize(coef_record.tail(num_exogen), dim);
	}

private:
	int dim, dim_exogen, nrow_exogen, num_exogen;
	Eigen::MatrixXd coef_mat;
};

/**
 * @brief Forecast class for `McmcTriangular`
 * 
 */
class CtaForecaster : public BayesForecaster<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	CtaForecaster(
		const RegRecords& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean,
		bool filter_stable, unsigned int seed, bool sv = true,
		BVHAR_OPTIONAL<std::unique_ptr<CtaExogenForecaster>> exogen_forecaster = BVHAR_NULLOPT
	)
	: BayesForecaster<Eigen::MatrixXd, Eigen::VectorXd>(step, response_mat, ord, records.coef_record.rows(), seed),
		include_mean(include_mean), stable_filter(filter_stable),
		dim(response.cols()),
		dim_design(include_mean ? lag * dim + 1 : lag * dim),
		num_coef(records.coef_record.cols()),
		// num_coef(dim * dim_design),
		// num_alpha(include_mean ? num_coef - dim : num_coef), nrow_coef(num_alpha / dim),
		num_alpha(include_mean ? num_coef - dim : num_coef),
		sv_update(Eigen::VectorXd::Zero(dim)),
		// coef_mat(Eigen::MatrixXd::Zero(num_coef / dim, dim)),
		contem_mat(Eigen::MatrixXd::Identity(dim, dim)),
		standard_normal(Eigen::VectorXd::Zero(dim)) {
		BVHAR_DEBUG_LOG(debug_logger, "CtaForecaster Constructor: step={}, ord={}, include_mean={}", step, ord, include_mean);
		initLagged();
		if (exogen_forecaster) {
			exogen_updater = std::move(*exogen_forecaster);
			num_coef -= exogen_updater->getSize();
			num_alpha -= exogen_updater->getSize();
		}
		nrow_coef = num_alpha / dim;
		coef_mat = Eigen::MatrixXd::Zero(num_coef / dim, dim);
	}
	virtual ~CtaForecaster() = default;

	/**
	 * @brief Draw density forecast
	 * 
	 * @return Eigen::MatrixXd Every forecast draw of which the row indicates forecast step and columns are blocked by chains.
	 */
	Eigen::MatrixXd forecastDensity() {
		return this->doForecast();
	}

	/**
	 * @copydoc forecastDensity()
	 * 
	 * @param valid_vec Validation vector to compute average log predictive likelihood (ALPL)
	 */
	Eigen::MatrixXd forecastDensity(const Eigen::VectorXd& valid_vec) {
		return this->doForecast(valid_vec);
	}

	Eigen::VectorXd getLastForecast() override {
		return this->doForecast().bottomRows<1>();
	}

	Eigen::VectorXd getLastForecast(const Eigen::VectorXd& valid_vec) override {
		return this->doForecast(valid_vec).bottomRows<1>();
	}

protected:
	std::unique_ptr<RegRecords> reg_record;
	std::unique_ptr<CtaExogenForecaster> exogen_updater;
	bool include_mean;
	bool stable_filter;
	int dim;
	int dim_design;
	int num_coef;
	int num_alpha;
	int nrow_coef; // dim_design in VAR and dim_har in VHAR (without constant term)
	Eigen::VectorXd sv_update; // d_1, ..., d_m
	Eigen::MatrixXd coef_mat; // include constant term when include_mean = true
	Eigen::MatrixXd contem_mat; // L
	Eigen::VectorXd standard_normal; // Z ~ N(0, I)

	void initLagged() override {
		BVHAR_DEBUG_LOG(debug_logger, "initLagged() called");
		last_pvec = Eigen::VectorXd::Zero(dim_design);
		point_forecast = Eigen::VectorXd::Zero(dim);
		pred_save = Eigen::MatrixXd::Zero(step, num_sim * dim);
		tmp_vec = Eigen::VectorXd::Zero((lag - 1) * dim);
		last_pvec[dim_design - 1] = 1.0; // valid when include_mean = true
		last_pvec.head(lag * dim) = vectorize_eigen(response.colwise().reverse().topRows(lag).transpose().eval()); // [y_T^T, y_(T - 1)^T, ... y_(T - lag + 1)^T]
	}

	void initRecursion(const Eigen::VectorXd& obs_vec) override {
		BVHAR_DEBUG_LOG(debug_logger, "initRecursion(obs_vec) called");
		last_pvec = obs_vec;
		point_forecast = obs_vec.head(dim);
		tmp_vec = obs_vec.segment(dim, (lag - 1) * dim);
	}

	void setRecursion() override {
		BVHAR_DEBUG_LOG(debug_logger, "setRecursion() called");
		last_pvec.segment(dim, (lag - 1) * dim) = tmp_vec;
		last_pvec.head(dim) = point_forecast;
	}

	void updatePred(const int h, const int i) override {
		BVHAR_DEBUG_LOG(debug_logger, "updatePred(h={}, i={}) called", h, i);
		computeMean();
		updateVariance();
		if (exogen_updater) {
			exogen_updater->appendForecast(point_forecast, h);
		}
		point_forecast += contem_mat.triangularView<Eigen::UnitLower>().solve(standard_normal); // N(point_forecast, L^-1 D L^T-1)
		pred_save.block(h, i * dim, 1, dim) = point_forecast.transpose(); // hat(Y_{T + h}^{(i)})
	}

	void updateRecursion() override {
		BVHAR_DEBUG_LOG(debug_logger, "updateRecursion() called");
		tmp_vec = last_pvec.head((lag - 1) * dim);
	}

	void forecastIn(const int i, const Eigen::MatrixXd& design) override {
		Eigen::MatrixXd point_pred = design.leftCols(num_coef / dim) * coef_mat;
		if (exogen_updater) {
			exogen_updater->appendPredict(point_pred, design);
		}
		pred_save.middleCols(i * dim, dim) = point_pred;
	}

	/**
	 * @brief Compute Normal mean of the forecast density
	 * 
	 */
	virtual void computeMean() = 0;

	/**
	 * @brief Draw innovation with D covariance matrix
	 * 
	 */
	virtual void updateVariance() = 0;
};

/**
 * @brief Forecast class for `McmcReg`
 * 
 */
class RegForecaster : public CtaForecaster {
public:
	RegForecaster(
		const LdltRecords& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean,
		bool filter_stable, unsigned int seed, bool sv = true,
		BVHAR_OPTIONAL<std::unique_ptr<CtaExogenForecaster>> exogen_forecaster = BVHAR_NULLOPT
	)
	: CtaForecaster(records, step, response_mat, ord, include_mean, filter_stable, seed, sv, std::move(exogen_forecaster)) {
		BVHAR_DEBUG_LOG(debug_logger, "RegForecaster Constructor: step={}, ord={}, include_mean={}", step, ord, include_mean);
		reg_record = std::make_unique<LdltRecords>(records);
	}
	virtual ~RegForecaster() = default;

protected:
	void updateParams(const int i) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateParams(i={}) called", i);
		coef_mat.topRows(nrow_coef) = unvectorize(reg_record->coef_record.row(i).head(num_alpha).transpose(), dim);
		if (include_mean) {
			coef_mat.bottomRows<1>() = reg_record->coef_record.row(i).segment(num_alpha, dim);
			// coef_mat.middleRows<1>(nrow_coef) = reg_record->coef_record.row(i).segment(num_alpha, dim);
		}
		if (exogen_updater) {
			exogen_updater->updateCoefmat(reg_record->coef_record.row(i).transpose());
		}
		reg_record->updateDiag(i, sv_update); // D^1/2
		contem_mat = build_inv_lower(dim, reg_record->contem_coef_record.row(i)); // L
	}
	void updateVariance() override {
		BVHAR_DEBUG_LOG(debug_logger, "updateVariance() called");
		for (int j = 0; j < dim; ++j) {
			standard_normal[j] = normal_rand(rng);
		}
		standard_normal.array() *= sv_update.array(); // D^(1/2) Z ~ N(0, D)
	}
	void updateLpl(int h, const Eigen::VectorXd& valid_vec) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateLpl(h={}, valid_vec) called", h);
		lpl[h] += sv_update.array().log().sum() - dim * log(2 * M_PI) / 2 - sv_update.cwiseInverse().cwiseProduct(contem_mat * (point_forecast - valid_vec)).squaredNorm() / 2;
	}
};

/**
 * @brief Forecast class for `McmcSv`
 * 
 */
class SvForecaster : public CtaForecaster {
public:
	SvForecaster(
		const SvRecords& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean,
		bool filter_stable, unsigned int seed, bool sv,
		BVHAR_OPTIONAL<std::unique_ptr<CtaExogenForecaster>> exogen_forecaster = BVHAR_NULLOPT
	)
	: CtaForecaster(records, step, response_mat, ord, include_mean, filter_stable, seed, sv, std::move(exogen_forecaster)),
		sv(sv), sv_sig(Eigen::VectorXd::Zero(dim)) {
		BVHAR_DEBUG_LOG(debug_logger, "SvForecaster Constructor: step={}, ord={}, include_mean={}", step, ord, include_mean);
		reg_record = std::make_unique<SvRecords>(records);
	}
	virtual ~SvForecaster() = default;

protected:
	void updateParams(const int i) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateParams(i={}) called", i);
		coef_mat.topRows(nrow_coef) = unvectorize(reg_record->coef_record.row(i).head(num_alpha).transpose(), dim);
		if (include_mean) {
			coef_mat.bottomRows<1>() = reg_record->coef_record.row(i).segment(num_alpha, dim);
		}
		if (exogen_updater) {
			exogen_updater->updateCoefmat(reg_record->coef_record.row(i).transpose());
		}
		reg_record->updateDiag(i, sv_update, sv_sig); // D^1/2
		contem_mat = build_inv_lower(dim, reg_record->contem_coef_record.row(i)); // L
	}
	void updateVariance() override {
		BVHAR_DEBUG_LOG(debug_logger, "updateVariance() called");
		if (sv) {
			for (int j = 0; j < dim; j++) {
				standard_normal[j] = normal_rand(rng);
			}
			standard_normal.array() *= sv_sig.array(); // sig_h Z ~ N(0, sig_h^2)
			sv_update.array() += standard_normal.array();
		}
		for (int j = 0; j < dim; j++) {
			standard_normal[j] = normal_rand(rng);
		}
		standard_normal.array() *= (sv_update / 2).array().exp(); // D^(1/2) Z ~ N(0, D)
	}
	void updateLpl(int h, const Eigen::VectorXd& valid_vec) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateLpl(h={}, valid_vec) called", h);
		lpl[h] += sv_update.sum() / 2 - dim * log(2 * M_PI) / 2 - ((-sv_update / 2).array().exp() * (contem_mat * (point_forecast - valid_vec)).array()).matrix().squaredNorm() / 2;
	}

private:
	bool sv;
	Eigen::VectorXd sv_sig; // sig_h
};

/**
 * @brief Forecast class of Bayesian VAR based on `McmcTriangular`
 * 
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 */
template <typename BaseForecaster = RegForecaster>
class CtaVarForecaster : public BaseForecaster {
public:
	CtaVarForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true,
		BVHAR_OPTIONAL<std::unique_ptr<CtaExogenForecaster>> exogen_forecaster = BVHAR_NULLOPT
	)
	: BaseForecaster(records, step, response_mat, lag, include_mean, filter_stable, seed, sv, std::move(exogen_forecaster)) {
		BVHAR_DEBUG_LOG(debug_logger, "CtaVarForecaster Constructor: step={}, lag={}, include_mean={}", step, lag, include_mean);
		if (stable_filter) {
			reg_record->subsetStable(num_alpha, 1);
			num_sim = reg_record->coef_record.rows();
			if (num_sim == 0) {
				BVHAR_STOP("No stable MCMC draws");
			}
			// Resize pred_save
		}
	}
	virtual ~CtaVarForecaster() = default;

protected:
	using BaseForecaster::reg_record;
	using BaseForecaster::stable_filter;
	using BaseForecaster::num_alpha;
	using BaseForecaster::num_sim;
	using BaseForecaster::point_forecast;
	using BaseForecaster::coef_mat;
	using BaseForecaster::last_pvec;
	using BaseForecaster::debug_logger;

	void computeMean() override {
		BVHAR_DEBUG_LOG(debug_logger, "computeMean() called");
		point_forecast = coef_mat.transpose() * last_pvec;
	}

	Eigen::MatrixXd getDesign() override {
		BVHAR_DEBUG_LOG(debug_logger, "getDesign() called");
		if (this->exogen_updater) {
			return build_x0(this->response, this->exogen_updater->getExogen(), this->lag, this->exogen_updater->getLag(), this->include_mean);
		}
		return build_x0(this->response, this->lag, this->include_mean);
	}
};

/**
 * @brief Forecast class of Bayesian VHAR based on `McmcTriangular`
 * 
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 */
template <typename BaseForecaster = RegForecaster>
class CtaVharForecaster : public BaseForecaster {
public:
	CtaVharForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true,
		BVHAR_OPTIONAL<std::unique_ptr<CtaExogenForecaster>> exogen_forecaster = BVHAR_NULLOPT
	)
	: BaseForecaster(records, step, response_mat, month, include_mean, filter_stable, seed, sv, std::move(exogen_forecaster)), har_trans(har_trans) {
		BVHAR_DEBUG_LOG(debug_logger, "CtaVharForecaster Constructor: step={}, month={}, include_mean={}", step, month, include_mean);
		if (stable_filter) {
			reg_record->subsetStable(num_alpha, 1, har_trans.topLeftCorner(3 * dim, month * dim));
			num_sim = reg_record->coef_record.rows();
			if (num_sim == 0) {
				BVHAR_STOP("No stable MCMC draws");
			}
			// Resize pred_save
		}
	}
	virtual ~CtaVharForecaster() = default;
	
protected:
	using BaseForecaster::reg_record;
	using BaseForecaster::stable_filter;
	using BaseForecaster::dim;
	using BaseForecaster::num_alpha;
	using BaseForecaster::num_sim;
	using BaseForecaster::lag;
	using BaseForecaster::include_mean;
	using BaseForecaster::response;
	using BaseForecaster::point_forecast;
	using BaseForecaster::coef_mat;
	using BaseForecaster::last_pvec;
	using BaseForecaster::exogen_updater;
	using BaseForecaster::debug_logger;
	Eigen::MatrixXd har_trans;

	void computeMean() override {
		BVHAR_DEBUG_LOG(debug_logger, "computeMean() called");
		point_forecast = coef_mat.transpose() * har_trans * last_pvec;
	}

	Eigen::MatrixXd getDesign() override {
		if (exogen_updater) {
			int dim_design = include_mean ? lag * dim + 1 : lag * dim;
			int dim_har = include_mean ? 3 * dim + 1 : 3 * dim;
			int dim_exogen = (exogen_updater->getLag() + 1) * (exogen_updater->getExogen()).cols();
			Eigen::MatrixXd vhar_design(response.rows(), dim_har + dim_exogen);
			Eigen::MatrixXd var_design = build_x0(response, exogen_updater->getExogen(), lag, exogen_updater->getLag(), include_mean);
			vhar_design.leftCols(dim_har) = var_design.leftCols(dim_design) * har_trans.transpose();
			vhar_design.rightCols(dim_exogen) = var_design.rightCols(dim_exogen);
			return vhar_design;
		}
		return build_x0(response, lag, include_mean) * har_trans.transpose();
	}
};

/**
 * @brief Bayesian VAR forecast class with sparse draw induced by posterior summary
 * 
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 */
template <typename BaseForecaster = RegForecaster>
class CtaVarSelectForecaster : public CtaVarForecaster<BaseForecaster> {
public:
	CtaVarSelectForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		double level, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true,
		BVHAR_OPTIONAL<std::unique_ptr<CtaExogenForecaster>> exogen_forecaster = BVHAR_NULLOPT
	)
	: CtaVarForecaster<BaseForecaster>(records, step, response_mat, lag, include_mean, filter_stable, seed, sv, std::move(exogen_forecaster)),
		activity_graph(unvectorize(reg_record->computeActivity(level), dim)) {
		BVHAR_DEBUG_LOG(debug_logger, "CtaVarSelectForecaster Constructor: level={}, step={}, lag={}, include_mean={}", level, step, lag, include_mean);
	}
	CtaVarSelectForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		const Eigen::MatrixXd& selection, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true,
		BVHAR_OPTIONAL<std::unique_ptr<CtaExogenForecaster>> exogen_forecaster = BVHAR_NULLOPT
	)
	: CtaVarForecaster<BaseForecaster>(records, step, response_mat, lag, include_mean, filter_stable, seed, sv, std::move(exogen_forecaster)),
		activity_graph(selection) {
		BVHAR_DEBUG_LOG(debug_logger, "CtaVarSelectForecaster Constructor: step={}, lag={}, include_mean={}", step, lag, include_mean);
	}
	
	virtual ~CtaVarSelectForecaster() = default;

protected:
	using CtaVarForecaster<BaseForecaster>::dim;
	using CtaVarForecaster<BaseForecaster>::reg_record;
	using CtaVarForecaster<BaseForecaster>::point_forecast;
	using CtaVarForecaster<BaseForecaster>::coef_mat;
	using CtaVarForecaster<BaseForecaster>::last_pvec;
	using CtaVarForecaster<BaseForecaster>::debug_logger;

	void computeMean() override {
		BVHAR_DEBUG_LOG(debug_logger, "computeMean() called");
		point_forecast = last_pvec.transpose() * (activity_graph.array() * coef_mat.array()).matrix();
	}

private:
	Eigen::MatrixXd activity_graph; // Activity graph computed after MCMC
};

/**
 * @brief Bayesian VHAR forecast class with sparse draw induced by posterior summary
 * 
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 */
template <typename BaseForecaster = RegForecaster>
class CtaVharSelectForecaster : public CtaVharForecaster<BaseForecaster> {
public:
	CtaVharSelectForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		double level, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true,
		BVHAR_OPTIONAL<std::unique_ptr<CtaExogenForecaster>> exogen_forecaster = BVHAR_NULLOPT
	)
	: CtaVharForecaster<BaseForecaster>(records, step, response_mat, har_trans, month, include_mean, filter_stable, seed, sv, std::move(exogen_forecaster)),
		activity_graph(unvectorize(reg_record->computeActivity(level), dim)) {
		BVHAR_DEBUG_LOG(debug_logger, "CtaVharSelectForecaster Constructor: level={}, step={}, month={}, include_mean={}", level, step, month, include_mean);
	}
	CtaVharSelectForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		const Eigen::MatrixXd& selection, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true,
		BVHAR_OPTIONAL<std::unique_ptr<CtaExogenForecaster>> exogen_forecaster = BVHAR_NULLOPT
	)
	: CtaVharForecaster<BaseForecaster>(records, step, response_mat, har_trans, month, include_mean, filter_stable, seed, sv, std::move(exogen_forecaster)),
		activity_graph(selection) {
		BVHAR_DEBUG_LOG(debug_logger, "CtaVharSelectForecaster Constructor: step={}, month={}, include_mean={}", step, month, include_mean);
	}
	
	virtual ~CtaVharSelectForecaster() = default;

protected:
	using CtaVharForecaster<BaseForecaster>::dim;
	using CtaVharForecaster<BaseForecaster>::reg_record;
	using CtaVharForecaster<BaseForecaster>::point_forecast;
	using CtaVharForecaster<BaseForecaster>::coef_mat;
	using CtaVharForecaster<BaseForecaster>::last_pvec;
	using CtaVharForecaster<BaseForecaster>::har_trans;
	using CtaVharForecaster<BaseForecaster>::debug_logger;

	void computeMean() override {
		BVHAR_DEBUG_LOG(debug_logger, "computeMean() called");
		point_forecast = last_pvec.transpose() * har_trans.transpose() * (activity_graph.array() * coef_mat.array()).matrix();
	}

private:
	Eigen::MatrixXd activity_graph; // Activity graph computed after MCMC
};

/**
 * @brief Initialize the vector of forecast class smart pointer
 * 
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 * @param num_chains Number of MCMC chains
 * @param ord VAR lag or VHAR month
 * @param step Forecasting step
 * @param response_mat Response matrix of multivariate regression
 * @param sparse If `true`, use sparsified records
 * @param level CI level
 * @param fit_record `BVHAR_LIST` of MCMC draws
 * @param seed_chain Random seed for each chain
 * @param include_mean If `true`, include constant term
 * @param stable If `true`, filter stable draws
 * @param nthreads Number of OpenMP threads
 * @param sv Use stochastic volaility when forecasting
 * @param har_trans VHAR transformation matrix
 * @return std::vector<std::unique_ptr<BaseForecaster>> Vector of forecast class smart pointer corresponding to each chain
 */
template <typename BaseForecaster = RegForecaster>
inline std::vector<std::unique_ptr<BaseForecaster>> initialize_ctaforecaster(
	int num_chains, int ord, int step, const Eigen::MatrixXd& response_mat,
	bool sparse, double level, BVHAR_LIST& fit_record,
	Eigen::Ref<const Eigen::VectorXi> seed_chain, bool include_mean, bool stable, int nthreads,
	bool sv = true, BVHAR_OPTIONAL<Eigen::MatrixXd> har_trans = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
) {
	bool activity = (level > 0); // BVHAR_OPTIONAL<double> level = BVHAR_NULLOPT
	if (sparse && activity) {
		BVHAR_STOP("If 'level > 0', 'sparse' should be false.");
	}
	using Records = typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type;
	std::vector<std::unique_ptr<BaseForecaster>> forecaster_ptr(num_chains);
	BVHAR_STRING coef_name = har_trans ? (sparse ? "phi_sparse_record" : "phi_record") : (sparse ? "alpha_sparse_record" : "alpha_record");
	BVHAR_STRING a_name = sparse ? "a_sparse_record" : "a_record";
	BVHAR_STRING c_name = sparse ? "c_sparse_record" : "c_record";
	for (int i = 0; i < num_chains; ++i) {
		std::unique_ptr<Records> reg_record;
		if (exogen) {
			BVHAR_STRING b_name = sparse ? "b_sparse_record" : "b_record";
			initialize_record(reg_record, i, fit_record, include_mean, coef_name, a_name, c_name, b_name);
		} else {
			initialize_record(reg_record, i, fit_record, include_mean, coef_name, a_name, c_name);
		}
		BVHAR_OPTIONAL<std::unique_ptr<CtaExogenForecaster>> exogen_updater = BVHAR_NULLOPT;
		if (exogen) {
			exogen_updater = std::make_unique<CtaExogenForecaster>(*exogen_lag, *exogen, response_mat.cols());
		}
		// std::unique_ptr<CtaExogenForecaster> exogen_updater;
		// if (exogen) {
		// 	exogen_updater = std::make_unique<CtaExogenForecaster>(*exogen_lag, *exogen, response_mat.cols());
		// } else {
		// 	exogen_updater = nullptr;
		// }
		if (har_trans && !activity) {
			forecaster_ptr[i] = std::make_unique<CtaVharForecaster<BaseForecaster>>(
				*reg_record, step, response_mat,
				*har_trans, ord,
				include_mean, stable, static_cast<unsigned int>(seed_chain[i]),
				sv, std::move(exogen_updater)
			);
		} else if (!har_trans && !activity) {
			forecaster_ptr[i] = std::make_unique<CtaVarForecaster<BaseForecaster>>(
				*reg_record, step, response_mat,
				ord,
				include_mean, stable, static_cast<unsigned int>(seed_chain[i]),
				sv, std::move(exogen_updater)
			);
		} else if (har_trans && activity) {
			forecaster_ptr[i] = std::make_unique<CtaVharSelectForecaster<BaseForecaster>>(
				*reg_record, level, step, response_mat,
				*har_trans, ord,
				include_mean, stable, static_cast<unsigned int>(seed_chain[i]),
				sv, std::move(exogen_updater)
			);
		} else {
			forecaster_ptr[i] = std::make_unique<CtaVarSelectForecaster<BaseForecaster>>(
				*reg_record, level, step, response_mat,
				ord,
				include_mean, stable, static_cast<unsigned int>(seed_chain[i]),
				sv, std::move(exogen_updater)
			);
		}
	}
	return forecaster_ptr;
}

/**
 * @brief CTA forecasting class
 * 
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 */
template <typename BaseForecaster = RegForecaster>
class CtaForecastRun : public McmcForecastRun<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	CtaForecastRun(
		int num_chains, int lag, int step, const Eigen::MatrixXd& response_mat,
		bool sparse, double level, BVHAR_LIST& fit_record,
		const Eigen::VectorXi& seed_chain, bool include_mean, bool stable, int nthreads,
		bool sv = true,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: McmcForecastRun<Eigen::MatrixXd, Eigen::VectorXd>(num_chains, lag, step, nthreads) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"CtaForecastRun Constructor: num_chains={}, lag={}, step={}, sparse={}, include_mean={}, nthreads={}",
			num_chains, lag, step, sparse, include_mean, nthreads
		);
		auto temp_forecaster = initialize_ctaforecaster<BaseForecaster>(
			num_chains, lag, step, response_mat, sparse, level,
			fit_record, seed_chain, include_mean,
			stable, nthreads, sv, BVHAR_NULLOPT, exogen, exogen_lag
		);
		for (int i = 0; i < num_chains; ++i) {
			forecaster[i] = std::move(temp_forecaster[i]);
		}
	}
	CtaForecastRun(
		int num_chains, int week, int month, int step, const Eigen::MatrixXd& response_mat,
		bool sparse, double level, BVHAR_LIST& fit_record,
		const Eigen::VectorXi& seed_chain, bool include_mean, bool stable, int nthreads,
		bool sv = true,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: McmcForecastRun<Eigen::MatrixXd, Eigen::VectorXd>(num_chains, month, step, nthreads) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"CtaForecastRun Constructor: num_chains={}, week={}, month={}, step={}, sparse={}, include_mean={}, nthreads={}",
			num_chains, week, month, step, sparse, include_mean, nthreads
		);
		Eigen::MatrixXd har_trans = build_vhar(response_mat.cols(), week, month, include_mean);
		auto temp_forecaster = initialize_ctaforecaster<BaseForecaster>(
			num_chains, month, step, response_mat, sparse, level,
			fit_record, seed_chain, include_mean,
			stable, nthreads, sv, har_trans, exogen, exogen_lag
		);
		for (int i = 0; i < num_chains; ++i) {
			forecaster[i] = std::move(temp_forecaster[i]);
		}
	}
	CtaForecastRun(
		int num_chains, int month, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans,
		bool sparse, double level, BVHAR_LIST& fit_record,
		const Eigen::VectorXi& seed_chain, bool include_mean, bool stable, int nthreads,
		bool sv = true,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: McmcForecastRun<Eigen::MatrixXd, Eigen::VectorXd>(num_chains, month, step, nthreads) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"CtaForecastRun Constructor: num_chains={}, month={}, step={}, sparse={}, include_mean={}, nthreads={}",
			num_chains, month, step, sparse, include_mean, nthreads
		);
		auto temp_forecaster = initialize_ctaforecaster<BaseForecaster>(
			num_chains, month, step, response_mat, sparse, level,
			fit_record, seed_chain, include_mean,
			stable, nthreads, sv, har_trans, exogen, exogen_lag
		);
		for (int i = 0; i < num_chains; ++i) {
			forecaster[i] = std::move(temp_forecaster[i]);
		}
	}
	virtual ~CtaForecastRun() = default;
};

/**
 * @brief Interface class for Out-of-sample forecasting with MCMC
 * 
 * @tparam BaseForecaster 
 */
template <typename BaseForecaster = RegForecaster>
class CtaOutforecastInterface {
public:
	virtual ~CtaOutforecastInterface() = default;

	/**
	 * @brief Out-of-sample forecasting
	 * 
	 */
	virtual void forecast() = 0;
};

/**
 * @brief Out-of-sample forecasting class
 * 
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 * @tparam isUpdate MCMC again in the new window
 */
template <typename BaseForecaster = RegForecaster, bool isUpdate = true>
class CtaOutforecastRun : public McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate> {
public:
	CtaOutforecastRun(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_reg, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept, BVHAR_LIST_OF_LIST& param_init, int prior_type,
		BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads, bool sv = true,
		BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>(
			y.rows(), lag,
			num_chains, num_iter, num_burn, thin, step, y_test, y_test.rows(), get_lpl, use_fit,
			seed_chain, seed_forecast, display_progress, nthreads,
			exogen_lag
		),
		dim(y.cols()), include_mean(include_mean), stable_filter(stable), sparse(sparse), sv(sv), level(level) {
		BVHAR_DEBUG_LOG(
			debug_logger, "CtaOutforecastRun Constructor: prior_type={}, contem_prior_type={}",
			prior_type, contem_prior_type
		);
		if (level > 0) {
			sparse = false;
		}
	}
	virtual ~CtaOutforecastRun() = default;

protected:
	using BaseMcmc = typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, McmcReg, McmcSv>::type;
	using RecordType = typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type;
	int dim;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::num_window;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::num_test;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::num_horizon;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::step;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::lag;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::num_chains;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::num_iter;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::num_burn;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::thin;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::nthreads;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::get_lpl;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::use_fit;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::display_progress;
	bool include_mean, stable_filter, sparse, sv;
	double level;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::seed_forecast;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::roll_mat;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::roll_y0;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::y_test;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::model;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::forecaster;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::out_forecast;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::lpl_record;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::roll_exogen_mat;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::roll_exogen;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::lag_exogen;
	using McmcOutForecastRun<Eigen::MatrixXd, Eigen::VectorXd, isUpdate>::debug_logger;

	/**
	 * @brief Define input in each window
	 * 
	 * @param y Entire data including validation set
	 */
	virtual void initData(const Eigen::MatrixXd& y, BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT) = 0;

	/**
	 * @brief Initialize forecaster
	 * 
	 * @param fit_record MCMC draw `BVHAR_LIST`
	 */
	virtual void initForecaster(BVHAR_LIST& fit_record) = 0;

	/**
	 * @brief Initialize CTA
	 * 
	 * @param param_reg `BVHAR_LIST` of CTA hyperparameters
	 * @param param_prior `BVHAR_LIST` of shrinkage prior hyperparameters
	 * @param param_intercept `BVHAR_LIST` of Normal prior hyperparameters for constant term
	 * @param param_init `BVHAR_LIST_OF_LIST` for initial values
	 * @param prior_type Shrinkage prior number
	 * @param grp_id Minnesota group unique id
	 * @param own_id own-lag id
	 * @param cross_id cross-lag id
	 * @param grp_mat Minnesota group matrix
	 * @param seed_chain Random seed for each chain
	 */
	virtual void initMcmc(
		BVHAR_LIST& param_reg, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept, BVHAR_LIST_OF_LIST& param_init, int prior_type,
		BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		const Eigen::MatrixXi& seed_chain,
		BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT
	) = 0;

	/**
	 * @brief Define VAR or VHAR design matrix
	 * 
	 * @param window Window index
	 * @return Eigen::MatrixXd Design matrix
	 */
	virtual Eigen::MatrixXd buildDesign(int window) = 0;

	/**
	 * @brief Initialize every member of `CtaOutforecastRun`
	 * 
	 * @param y Response matrix
	 * @param fit_record `BVHAR_LIST` of MCMC draws
	 * @param param_reg `BVHAR_LIST` of CTA hyperparameters
	 * @param param_prior `BVHAR_LIST` of shrinkage prior hyperparameters
	 * @param param_intercept `BVHAR_LIST` of Normal prior hyperparameters for constant term
	 * @param param_init `BVHAR_LIST_OF_LIST` for initial values
	 * @param prior_type Shrinkage prior number
	 * @param grp_id Minnesota group unique id
	 * @param own_id own-lag id
	 * @param cross_id cross-lag id
	 * @param grp_mat Minnesota group matrix
	 * @param seed_chain Random seed for each chain
	 */
	void initialize(
		const Eigen::MatrixXd& y, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_reg, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept, BVHAR_LIST_OF_LIST& param_init, int prior_type,
		BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		const Eigen::MatrixXi& seed_chain,
		BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	) {
		BVHAR_DEBUG_LOG(debug_logger, "initialize(...) called");
		initData(y, exogen);
		if (use_fit) {
			initForecaster(fit_record);
		}
		using is_mcmc = std::integral_constant<bool, isUpdate>;
		if (is_mcmc::value) {
			initMcmc(
				param_reg, param_prior, param_intercept, param_init, prior_type,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, seed_chain,
				exogen_prior, exogen_init, exogen_prior_type
			);
		}
	}

	Eigen::VectorXd getValid() override {
		BVHAR_DEBUG_LOG(debug_logger, "getValid() called");
		return y_test.row(step);
	}
};

/**
 * @brief Rolling-window forecast class
 * 
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 * @tparam isGroup If `true`, use group shrinkage parameter
 * @tparam isUpdate MCMC again in the new window
 */
template <typename BaseForecaster = RegForecaster, bool isGroup = true, bool isUpdate = true>
class CtaRollforecastRun : public CtaOutforecastRun<BaseForecaster, isUpdate> {
public:
	CtaRollforecastRun(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_reg, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept, BVHAR_LIST_OF_LIST& param_init, int prior_type,
		BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads, bool sv = true,
		BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: CtaOutforecastRun<BaseForecaster, isUpdate>(
			y, lag, num_chains, num_iter, num_burn, thin, sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test, get_lpl, use_fit,
			seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		) {
		BVHAR_DEBUG_LOG(debug_logger, "CtaRollforecastRun Constructor");
	}
	virtual ~CtaRollforecastRun() = default;

protected:
	using typename CtaOutforecastRun<BaseForecaster, isUpdate>::BaseMcmc;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::num_window;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::dim;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::num_test;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::num_horizon;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::step;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::lag;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::num_chains;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::num_iter;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::num_burn;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::use_fit;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::include_mean;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::roll_mat;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::roll_y0;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::y_test;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::model;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::forecaster;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::buildDesign;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::initialize;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::roll_exogen_mat;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::roll_exogen;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::lag_exogen;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::debug_logger;

	void initData(const Eigen::MatrixXd& y, BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT) override {
		BVHAR_DEBUG_LOG(debug_logger, "initData(y, ...) called");
		Eigen::MatrixXd tot_mat(num_window + num_test, dim);
		tot_mat << y,
							 y_test;
		for (int i = 0; i < num_horizon; ++i) {
			roll_mat[i] = tot_mat.middleRows(i, num_window);
			roll_y0[i] = build_y0(roll_mat[i], lag, lag + 1);
		}
		if (lag_exogen) {
			for (int i = 0; i < num_horizon; ++i) {
				roll_exogen_mat[i] = (*exogen).middleRows(i, num_window);
				roll_exogen[i] = (*exogen).middleRows(num_window - *lag_exogen + i, *lag_exogen + step);
			}
		}
	}
	void initMcmc(
		BVHAR_LIST& param_reg, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept, BVHAR_LIST_OF_LIST& param_init, int prior_type,
		BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		const Eigen::MatrixXi& seed_chain,
		BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT
	) override {
		BVHAR_DEBUG_LOG(debug_logger, "initMcmc(...) called");
		BVHAR_OPTIONAL<int> exogen_cols = BVHAR_NULLOPT;
		for (int window = 0; window < num_horizon; ++window) {
			if (use_fit && window ==0) {
				continue;
			}
			Eigen::MatrixXd design = buildDesign(window);
			if (lag_exogen) {
				exogen_cols = (*lag_exogen + 1) * roll_exogen_mat[window]->cols();
			}
			auto temp_mcmc = initialize_mcmc<BaseMcmc, isGroup>(
				num_chains, num_iter - num_burn, design, roll_y0[window],
				param_reg, param_prior, param_intercept, param_init, prior_type,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain.row(window), BVHAR_NULLOPT,
				exogen_prior, exogen_init, exogen_prior_type, exogen_cols
			);
			for (int i = 0; i < num_chains; ++i) {
				model[window][i] = std::move(temp_mcmc[i]);
			}
			roll_mat[window].resize(0, 0); // free the memory
		}
	}
};

/**
 * @brief Expanding-window forecast class
 * 
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 * @tparam isGroup If `true`, use group shrinkage parameter
 * @tparam isUpdate MCMC again in the new window
 */
template <typename BaseForecaster = RegForecaster, bool isGroup = true, bool isUpdate = true>
class CtaExpandforecastRun : public CtaOutforecastRun<BaseForecaster, isUpdate> {
public:
	CtaExpandforecastRun(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_reg, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept, BVHAR_LIST_OF_LIST& param_init, int prior_type,
		BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads, bool sv = true,
		BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: CtaOutforecastRun<BaseForecaster, isUpdate>(
			y, lag, num_chains, num_iter, num_burn, thin, sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test, get_lpl, use_fit,
			seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		) {
		BVHAR_DEBUG_LOG(debug_logger, "CtaExpandforecastRun Constructor");
	}
	virtual ~CtaExpandforecastRun() = default;

protected:
	using typename CtaOutforecastRun<BaseForecaster, isUpdate>::BaseMcmc;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::num_window;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::dim;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::num_test;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::num_horizon;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::step;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::lag;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::num_chains;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::num_iter;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::num_burn;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::use_fit;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::include_mean;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::roll_mat;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::roll_y0;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::y_test;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::model;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::forecaster;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::buildDesign;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::initialize;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::roll_exogen_mat;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::roll_exogen;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::lag_exogen;
	using CtaOutforecastRun<BaseForecaster, isUpdate>::debug_logger;

	void initData(const Eigen::MatrixXd& y, BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT) override {
		BVHAR_DEBUG_LOG(debug_logger, "initData(y, ...) called");
		Eigen::MatrixXd tot_mat(num_window + num_test, dim);
		tot_mat << y,
							 y_test;
		for (int i = 0; i < num_horizon; ++i) {
			roll_mat[i] = tot_mat.topRows(num_window + i);
			roll_y0[i] = build_y0(roll_mat[i], lag, lag + 1);
		}
		if (lag_exogen && exogen) {
			for (int i = 0; i < num_horizon; ++i) {
				roll_exogen_mat[i] = (*exogen).topRows(num_window + i);
				roll_exogen[i] = (*exogen).middleRows(num_window - *lag_exogen + i, *lag_exogen + step);
			}
		}
	}
	void initMcmc(
		BVHAR_LIST& param_reg, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept, BVHAR_LIST_OF_LIST& param_init, int prior_type,
		BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		const Eigen::MatrixXi& seed_chain,
		BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT
	) override {
		BVHAR_DEBUG_LOG(debug_logger, "initMcmc(...) called");
		BVHAR_OPTIONAL<int> exogen_cols = BVHAR_NULLOPT;
		for (int window = 0; window < num_horizon; ++window) {
			if (use_fit && window ==0) {
				continue;
			}
			Eigen::MatrixXd design = buildDesign(window);
			if (lag_exogen) {
				exogen_cols = (*lag_exogen + 1) * roll_exogen[window]->cols();
			}
			if (BVHAR_CONTAINS(param_reg, "initial_mean")) {
				// BaseMcmc == McmcSv
				auto temp_mcmc = initialize_mcmc<BaseMcmc, isGroup>(
					num_chains, num_iter - num_burn, design, roll_y0[window],
					param_reg, param_prior, param_intercept, param_init, prior_type,
					contem_prior, contem_init, contem_prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain.row(window),
					roll_y0[window].rows(),
					exogen_prior, exogen_init, exogen_prior_type, exogen_cols
				);
				for (int i = 0; i < num_chains; ++i) {
					model[window][i] = std::move(temp_mcmc[i]);
				}
			} else {
				// BaseMcmc == McmcReg
				auto temp_mcmc = initialize_mcmc<BaseMcmc, isGroup>(
					num_chains, num_iter - num_burn, design, roll_y0[window],
					param_reg, param_prior, param_intercept, param_init, prior_type,
					contem_prior, contem_init, contem_prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain.row(window), BVHAR_NULLOPT,
					exogen_prior, exogen_init, exogen_prior_type, exogen_cols
				);
				for (int i = 0; i < num_chains; ++i) {
					model[window][i] = std::move(temp_mcmc[i]);
				}
			}
			roll_mat[window].resize(0, 0); // free the memory
		}
	}
};

/**
 * @brief Out-of-sample forecast class for Bayesian VAR
 * 
 * @tparam BaseOutForecast `CtaRollforecastRun` or `CtaExpandforecastRun`
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 * @tparam isGroup If `true`, use group shrinkage parameter
 * @tparam isUpdate MCMC again in the new window
 */
template <template <typename, bool, bool> class BaseOutForecast = CtaRollforecastRun, typename BaseForecaster = RegForecaster, bool isGroup = true, bool isUpdate = true>
class CtaVarforecastRun : public BaseOutForecast<BaseForecaster, isGroup, isUpdate> {
public:
	CtaVarforecastRun(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_reg, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept, BVHAR_LIST_OF_LIST& param_init, int prior_type,
		BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads, bool sv = true,
		BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: BaseOutForecast<BaseForecaster, isGroup, isUpdate>(
			y, lag, num_chains, num_iter, num_burn, thin, sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test, get_lpl, use_fit,
			seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		) {
		BVHAR_DEBUG_LOG(debug_logger, "CtaVarforecastRun Constructor");
		initialize(
			y, fit_record, param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, seed_chain,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		);
	}
	virtual ~CtaVarforecastRun() = default;

protected:
	using typename BaseOutForecast<BaseForecaster, isGroup, isUpdate>::BaseMcmc;
	using typename BaseOutForecast<BaseForecaster, isGroup, isUpdate>::RecordType;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::dim;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::num_horizon;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::step;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::lag;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::num_chains;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::num_iter;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::num_burn;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::thin;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::nthreads;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::include_mean;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::stable_filter;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::sparse;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::sv;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::level;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::seed_forecast;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::roll_mat;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::roll_y0;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::model;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::forecaster;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::out_forecast;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::lpl_record;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::initialize;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::roll_exogen_mat;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::roll_exogen;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::lag_exogen;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::debug_logger;

	void initForecaster(BVHAR_LIST& fit_record) override {
		BVHAR_DEBUG_LOG(debug_logger, "initForecaster(fit_record) called");
		using is_mcmc = std::integral_constant<bool, isUpdate>;
		if (is_mcmc::value) {
			auto temp_forecaster = initialize_ctaforecaster<BaseForecaster>(
				num_chains, lag, step, roll_y0[0], sparse, level,
				fit_record, seed_forecast, include_mean,
				stable_filter, nthreads, sv, BVHAR_NULLOPT,
				roll_exogen[0], lag_exogen
			);
			for (int i = 0; i < num_chains; ++i) {
				forecaster[0][i] = std::move(temp_forecaster[i]);
			}
		} else {
			for (int i = 0; i < num_horizon; ++i) {
				auto temp_forecaster = initialize_ctaforecaster<BaseForecaster>(
					num_chains, lag, step, roll_y0[i], sparse, level,
					fit_record, seed_forecast, include_mean,
					stable_filter, nthreads, sv, BVHAR_NULLOPT,
					roll_exogen[i], lag_exogen
				);
				for (int j = 0; j < num_chains; ++j) {
					forecaster[i][j] = std::move(temp_forecaster[j]);
				}
			}
		}
	}

	Eigen::MatrixXd buildDesign(int window) override {
		BVHAR_DEBUG_LOG(debug_logger, "buildDesign(window={}) called", window);
		if (lag_exogen) {
			return build_x0(roll_mat[window], *(roll_exogen_mat[window]), lag, *lag_exogen, include_mean);
		}
		return build_x0(roll_mat[window], lag, include_mean);
	}

	void updateForecaster(int window, int chain) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateForecaster(window={}, chain={}) called", window, chain);
		// RecordType reg_record = model[window][chain]->template returnStructRecords<RecordType>(0, thin, sparse);
		auto* mcmc_triangular = dynamic_cast<McmcTriangular*>(model[window][chain].get());
		if (!mcmc_triangular) {
			BVHAR_STOP("Model is not a McmcTriangular.");
		}
		RecordType reg_record = mcmc_triangular->template returnStructRecords<RecordType>(0, thin, sparse);
		BVHAR_OPTIONAL<std::unique_ptr<CtaExogenForecaster>> exogen_updater = BVHAR_NULLOPT;
		if (lag_exogen) {
			exogen_updater = std::make_unique<CtaExogenForecaster>(*lag_exogen, *(roll_exogen[window]), dim);
		}
		if (level > 0) {
			forecaster[window][chain] = std::make_unique<CtaVarSelectForecaster<BaseForecaster>>(
				reg_record, level, step, roll_y0[window], lag, include_mean,
				stable_filter, static_cast<unsigned int>(seed_forecast[chain]), sv,
				std::move(exogen_updater)
			);
		} else {
			forecaster[window][chain] = std::make_unique<CtaVarForecaster<BaseForecaster>>(
				reg_record, step, roll_y0[window], lag, include_mean,
				stable_filter, static_cast<unsigned int>(seed_forecast[chain]), sv,
				std::move(exogen_updater)
			);
		}
		model[window][chain].reset();
	}
};

/**
 * @brief Out-of-sample forecast class for Bayesian VHAR
 * 
 * @tparam BaseOutForecast `CtaRollforecastRun` or `CtaExpandforecastRun`
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 * @tparam isGroup If `true`, use group shrinkage parameter
 * @tparam isUpdate MCMC again in the new window
 */
template <template <typename, bool, bool> class BaseOutForecast = CtaRollforecastRun, typename BaseForecaster = RegForecaster, bool isGroup = true, bool isUpdate = true>
class CtaVharforecastRun : public BaseOutForecast<BaseForecaster, isGroup, isUpdate> {
public:
	CtaVharforecastRun(
		const Eigen::MatrixXd& y, int week, int month, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_reg, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept, BVHAR_LIST_OF_LIST& param_init, int prior_type,
		BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads, bool sv = true,
		BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
	)
	: BaseOutForecast<BaseForecaster, isGroup, isUpdate>(
			y, month, num_chains, num_iter, num_burn, thin, sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test, get_lpl, use_fit,
			seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		),
		har_trans(build_vhar(dim, week, month, include_mean)) {
		BVHAR_DEBUG_LOG(debug_logger, "CtaVharforecastRun Constructor");
		initialize(
			y, fit_record, param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, seed_chain,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		);
	}
	virtual ~CtaVharforecastRun() = default;

protected:
	using typename BaseOutForecast<BaseForecaster, isGroup, isUpdate>::BaseMcmc;
	using typename BaseOutForecast<BaseForecaster, isGroup, isUpdate>::RecordType;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::dim;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::num_horizon;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::step;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::lag;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::num_chains;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::num_iter;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::num_burn;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::thin;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::nthreads;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::include_mean;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::stable_filter;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::sparse;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::sv;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::level;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::seed_forecast;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::roll_mat;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::roll_y0;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::model;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::forecaster;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::out_forecast;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::lpl_record;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::initialize;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::roll_exogen_mat;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::roll_exogen;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::lag_exogen;
	using BaseOutForecast<BaseForecaster, isGroup, isUpdate>::debug_logger;
	Eigen::MatrixXd har_trans;

	void initForecaster(BVHAR_LIST& fit_record) override {
		BVHAR_DEBUG_LOG(debug_logger, "initForecaster(fit_record) called");
		using is_mcmc = std::integral_constant<bool, isUpdate>;
		if (is_mcmc::value) {
			auto temp_forecaster = initialize_ctaforecaster<BaseForecaster>(
				num_chains, lag, step, roll_y0[0], sparse, level,
				fit_record, seed_forecast, include_mean,
				stable_filter, nthreads, sv,
				har_trans,
				roll_exogen[0], lag_exogen
			);
			for (int i = 0; i < num_chains; ++i) {
				forecaster[0][i] = std::move(temp_forecaster[i]);
			}
		} else {
			for (int i = 0; i < num_horizon; ++i) {
				auto temp_forecaster = initialize_ctaforecaster<BaseForecaster>(
					num_chains, lag, step, roll_y0[i], sparse, level,
					fit_record, seed_forecast, include_mean,
					stable_filter, nthreads, sv,
					har_trans,
					roll_exogen[i], lag_exogen
				);
				for (int j = 0; j < num_chains; ++j) {
					forecaster[i][j] = std::move(temp_forecaster[j]);
				}
			}
		}
	}

	Eigen::MatrixXd buildDesign(int window) override {
		BVHAR_DEBUG_LOG(debug_logger, "buildDesign(window={}) called", window);
		if (lag_exogen) {
			int dim_design = include_mean ? lag * dim + 1 : lag * dim;
			int dim_har = include_mean ? 3 * dim + 1 : 3 * dim;
			int dim_exogen = (*lag_exogen + 1) * roll_exogen_mat[window]->cols();
			Eigen::MatrixXd vhar_design(roll_y0[window].rows(), dim_har + dim_exogen);
			Eigen::MatrixXd var_design = build_x0(roll_mat[window], *(roll_exogen_mat[window]), lag, *lag_exogen, include_mean);
			vhar_design.leftCols(dim_har) = var_design.leftCols(dim_design) * har_trans.transpose();
			vhar_design.rightCols(dim_exogen) = var_design.rightCols(dim_exogen);
			return vhar_design;
		}
		return build_x0(roll_mat[window], lag, include_mean) * har_trans.transpose();
	}

	void updateForecaster(int window, int chain) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateForecaster(window={}, chain={}) called", window, chain);
		// RecordType reg_record = model[window][chain]->template returnStructRecords<RecordType>(0, thin, sparse);
		auto* mcmc_triangular = dynamic_cast<McmcTriangular*>(model[window][chain].get());
		if (!mcmc_triangular) {
			BVHAR_STOP("Model is not a McmcTriangular.");
		}
		RecordType reg_record = mcmc_triangular->template returnStructRecords<RecordType>(0, thin, sparse);
		BVHAR_OPTIONAL<std::unique_ptr<CtaExogenForecaster>> exogen_updater = BVHAR_NULLOPT;
		if (lag_exogen) {
			exogen_updater = std::make_unique<CtaExogenForecaster>(*lag_exogen, *(roll_exogen[window]), dim);
		}
		if (level > 0) {
			forecaster[window][chain] = std::make_unique<CtaVharSelectForecaster<BaseForecaster>>(
				reg_record, level, step, roll_y0[window], har_trans, lag, include_mean,
				stable_filter, static_cast<unsigned int>(seed_forecast[chain]), sv,
				std::move(exogen_updater)
			);
		} else {
			forecaster[window][chain] = std::make_unique<CtaVharForecaster<BaseForecaster>>(
				reg_record, step, roll_y0[window], har_trans, lag, include_mean,
				stable_filter, static_cast<unsigned int>(seed_forecast[chain]), sv,
				std::move(exogen_updater)
			);
		}
		model[window][chain].reset();
	}
};

template <template <typename, bool, bool> class BaseOutForecast = CtaRollforecastRun, typename BaseForecaster = RegForecaster>
inline std::unique_ptr<McmcOutforecastInterface> initialize_ctaoutforecaster(
	const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
	bool sparse, double level, BVHAR_LIST& fit_record, bool run_mcmc,
	BVHAR_LIST& param_reg, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept, BVHAR_LIST_OF_LIST& param_init, int prior_type, bool ggl,
	BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
	const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
	bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test,
	bool get_lpl, bool use_fit, const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
	const bool sv,
	BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
) {
	if (ggl && run_mcmc) {
		return std::make_unique<CtaVarforecastRun<BaseOutForecast, BaseForecaster, true, true>>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		);
	} else if (ggl && !run_mcmc) {
		return std::make_unique<CtaVarforecastRun<BaseOutForecast, BaseForecaster, true, false>>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		);
	} else if (!ggl && run_mcmc) {
		return std::make_unique<CtaVarforecastRun<BaseOutForecast, BaseForecaster, false, true>>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		);
	}
	return std::make_unique<CtaVarforecastRun<BaseOutForecast, BaseForecaster, false, false>>(
		y, lag, num_chains, num_iter, num_burn, thinning,
		sparse, level, fit_record,
		param_reg, param_prior, param_intercept, param_init, prior_type,
		contem_prior, contem_init, contem_prior_type,
		grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
		get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
		exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
	);
}

template <template <typename, bool, bool> class BaseOutForecast = CtaRollforecastRun, typename BaseForecaster = RegForecaster>
inline std::unique_ptr<McmcOutforecastInterface> initialize_ctaoutforecaster(
	const Eigen::MatrixXd& y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
	bool sparse, double level, BVHAR_LIST& fit_record, bool run_mcmc,
	BVHAR_LIST& param_reg, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept, BVHAR_LIST_OF_LIST& param_init, int prior_type, bool ggl,
	BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
	const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
	bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test,
	bool get_lpl, bool use_fit, const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
	const bool sv,
	BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT
) {
	if (ggl && run_mcmc) {
		return std::make_unique<CtaVharforecastRun<BaseOutForecast, BaseForecaster, true, true>>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		);
	} else if (ggl && !run_mcmc) {
		return std::make_unique<CtaVharforecastRun<BaseOutForecast, BaseForecaster, true, false>>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		);
	} else if (!ggl && run_mcmc) {
		return std::make_unique<CtaVharforecastRun<BaseOutForecast, BaseForecaster, false, true>>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		);
	}
	return std::make_unique<CtaVharforecastRun<BaseOutForecast, BaseForecaster, false, false>>(
		y, week, month, num_chains, num_iter, num_burn, thinning,
		sparse, level, fit_record,
		param_reg, param_prior, param_intercept, param_init, prior_type,
		contem_prior, contem_init, contem_prior_type,
		grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
		get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
		exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
	);
}

} // namespace bvhar

#endif // BVHAR_BAYES_TRIANGULAR_FORECASTER_H