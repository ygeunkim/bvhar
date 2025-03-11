#ifndef BVHAR_BAYES_TRIANGULAR_FORECASTER_H
#define BVHAR_BAYES_TRIANGULAR_FORECASTER_H

#include "./triangular.h"

namespace bvhar {

class McmcForecaster;
class RegForecaster;
class SvForecaster;
template <typename BaseForecaster> class McmcVarForecaster;
template <typename BaseForecaster> class McmcVharForecaster;
template <typename BaseForecaster> class McmcVarSelectForecaster;
template <typename BaseForecaster> class McmcVharSelectForecaster;
// Running forecasters
template <typename BaseForecaster> class McmcForecastRun;
template <typename BaseForecaster> class McmcOutforecastInterface;
template <typename BaseForecaster, bool> class McmcOutforecastRun;
template <typename BaseForecaster, bool, bool> class McmcRollforecastRun;
template <typename BaseForecaster, bool, bool> class McmcExpandforecastRun;
template <template <typename, bool, bool> class BaseOutForecast, typename BaseForecaster, bool, bool> class McmcVarforecastRun;
template <template <typename, bool, bool> class BaseOutForecast, typename BaseForecaster, bool, bool> class McmcVharforecastRun;

/**
 * @brief Forecast class for `McmcTriangular`
 * 
 */
class McmcForecaster {
public:
	McmcForecaster(const RegRecords& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true)
	: rng(seed), response(response_mat), include_mean(include_mean), stable_filter(filter_stable),
		step(step), dim(response.cols()), var_lag(ord),
		dim_design(include_mean ? var_lag * dim + 1 : var_lag * dim),
		num_coef(records.coef_record.cols()),
		num_alpha(include_mean ? num_coef - dim : num_coef), nrow_coef(num_alpha / dim),
		num_sim(records.coef_record.rows()),
		last_pvec(Eigen::VectorXd::Zero(dim_design)),
		sv_update(Eigen::VectorXd::Zero(dim)),
		post_mean(Eigen::VectorXd::Zero(dim)),
		predictive_distn(Eigen::MatrixXd::Zero(step, num_sim * dim)),
		coef_mat(Eigen::MatrixXd::Zero(num_coef / dim, dim)),
		contem_mat(Eigen::MatrixXd::Identity(dim, dim)),
		standard_normal(Eigen::VectorXd::Zero(dim)),
		tmp_vec(Eigen::VectorXd::Zero((var_lag - 1) * dim)),
		lpl(Eigen::VectorXd::Zero(step)) {
		last_pvec[dim_design - 1] = 1.0; // valid when include_mean = true
		last_pvec.head(var_lag * dim) = vectorize_eigen(response.colwise().reverse().topRows(var_lag).transpose().eval()); // [y_T^T, y_(T - 1)^T, ... y_(T - lag + 1)^T]
	}
	virtual ~McmcForecaster() = default;

	/**
	 * @brief Draw density forecast
	 * 
	 * @return Eigen::MatrixXd Every forecast draw of which the row indicates forecast step and columns are blocked by chains.
	 */
	Eigen::MatrixXd forecastDensity() {
		std::lock_guard<std::mutex> lock(mtx);
		Eigen::VectorXd obs_vec = last_pvec; // y_T, y_(T - 1), ... y_(T - lag + 1)
		for (int i = 0; i < num_sim; ++i) {
			last_pvec = obs_vec; // y_T, y_(T - 1), ... y_(T - lag + 1)
			post_mean = obs_vec.head(dim); // y_T
			tmp_vec = obs_vec.segment(dim, (var_lag - 1) * dim); // y_(T - 1), ... y_(T - lag + 1)
			updateParams(i);
			forecastOut(i);
		}
		return predictive_distn;
	}

	/**
	 * @copydoc forecastDensity()
	 * 
	 * @param valid_vec Validation vector to compute average log predictive likelihood (ALPL)
	 */
	Eigen::MatrixXd forecastDensity(const Eigen::VectorXd& valid_vec) {
		std::lock_guard<std::mutex> lock(mtx);
		Eigen::VectorXd obs_vec = last_pvec; // y_T, y_(T - 1), ... y_(T - lag + 1)
		for (int i = 0; i < num_sim; ++i) {
			last_pvec = obs_vec;
			post_mean = obs_vec.head(dim);
			tmp_vec = obs_vec.segment(dim, (var_lag - 1) * dim);
			updateParams(i);
			forecastOut(i, valid_vec);
		}
		lpl.array() /= num_sim;
		return predictive_distn;
	}

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
	std::unique_ptr<RegRecords> reg_record;
	BHRNG rng;
	std::mutex mtx;
	Eigen::MatrixXd response; // y0
	bool include_mean;
	bool stable_filter;
	int step;
	int dim;
	int var_lag; // VAR order or month order of VHAR
	int dim_design;
	int num_coef;
	int num_alpha;
	int nrow_coef; // dim_design in VAR and dim_har in VHAR (without constant term)
	int num_sim;
	Eigen::VectorXd last_pvec; // [ y_(T + h - 1)^T, y_(T + h - 2)^T, ..., y_(T + h - p)^T, 1 ] (1 when constant term)
	Eigen::VectorXd sv_update; // d_1, ..., d_m
	Eigen::VectorXd post_mean; // posterior mean and y_(T + h - 1)
	Eigen::MatrixXd predictive_distn; // rbind(step), cbind(sims)
	Eigen::MatrixXd coef_mat; // include constant term when include_mean = true
	Eigen::MatrixXd contem_mat; // L
	Eigen::VectorXd standard_normal; // Z ~ N(0, I)
	Eigen::VectorXd tmp_vec; // y_(T + h - 2), ... y_(T + h - lag)
	Eigen::VectorXd lpl; // average log-predictive likelihood

	/**
	 * @brief Compute Normal mean of the forecast density
	 * 
	 */
	virtual void computeMean() = 0;

	/**
	 * @brief Update members with corresponding MCMC draw
	 * 
	 * @param i MCMC step
	 */
	virtual void updateParams(int i) = 0;

	/**
	 * @brief Draw innovation with D covariance matrix
	 * 
	 */
	virtual void updateVariance() = 0;

	/**
	 * @brief Compute LPL
	 * 
	 * @param h Forecast step
	 * @param valid_vec Validation vector
	 */
	virtual void updateLpl(int h, const Eigen::VectorXd& valid_vec) = 0;

	/**
	 * @brief Draw i-th forecast
	 * 
	 * @param i MCMC step
	 */
	void forecastOut(int i) {
		for (int h = 0; h < step; ++h) {
			last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
			last_pvec.head(dim) = post_mean;
			computeMean();
			updateVariance();
			post_mean += contem_mat.triangularView<Eigen::UnitLower>().solve(standard_normal); // N(post_mean, L^-1 D L^T-1)
			predictive_distn.block(h, i * dim, 1, dim) = post_mean.transpose(); // hat(Y_{T + h}^{(i)})
			tmp_vec = last_pvec.head((var_lag - 1) * dim);
		}
	}

	/**
	 * @copydoc forecastOut(int)
	 * 
	 * @param valid_vec Validation vector
	 */
	void forecastOut(int i, const Eigen::VectorXd& valid_vec) {
		for (int h = 0; h < step; ++h) {
			last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
			last_pvec.head(dim) = post_mean;
			computeMean();
			updateVariance();
			post_mean += contem_mat.triangularView<Eigen::UnitLower>().solve(standard_normal); // N(post_mean, L^-1 D L^T-1)
			predictive_distn.block(h, i * dim, 1, dim) = post_mean.transpose(); // hat(Y_{T + h}^{(i)})
			updateLpl(h, valid_vec);
			tmp_vec = last_pvec.head((var_lag - 1) * dim);
		}
	}
};

/**
 * @brief Forecast class for `McmcReg`
 * 
 */
class RegForecaster : public McmcForecaster {
public:
	RegForecaster(const LdltRecords& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true)
	: McmcForecaster(records, step, response_mat, ord, include_mean, filter_stable, seed, sv) {
		reg_record = std::make_unique<LdltRecords>(records);
	}
	virtual ~RegForecaster() = default;

protected:
	void updateParams(int i) override {
		coef_mat.topRows(nrow_coef) = unvectorize(reg_record->coef_record.row(i).head(num_alpha).transpose(), dim);
		if (include_mean) {
			coef_mat.bottomRows(1) = reg_record->coef_record.row(i).tail(dim);
		}
		reg_record->updateDiag(i, sv_update); // D^1/2
		contem_mat = build_inv_lower(dim, reg_record->contem_coef_record.row(i)); // L
	}
	void updateVariance() override {
		for (int j = 0; j < dim; ++j) {
			standard_normal[j] = normal_rand(rng);
		}
		standard_normal.array() *= sv_update.array(); // D^(1/2) Z ~ N(0, D)
	}
	void updateLpl(int h, const Eigen::VectorXd& valid_vec) override {
		lpl[h] += sv_update.array().log().sum() - dim * log(2 * M_PI) / 2 - sv_update.cwiseInverse().cwiseProduct(contem_mat * (post_mean - valid_vec)).squaredNorm() / 2;
	}
};

/**
 * @brief Forecast class for `McmcSv`
 * 
 */
class SvForecaster : public McmcForecaster {
public:
	SvForecaster(const SvRecords& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean, bool filter_stable, unsigned int seed, bool sv)
	: McmcForecaster(records, step, response_mat, ord, include_mean, filter_stable, seed, sv),
		sv(sv), sv_sig(Eigen::VectorXd::Zero(dim)) {
		reg_record = std::make_unique<SvRecords>(records);
	}
	virtual ~SvForecaster() = default;

protected:
	void updateParams(int i) override {
		coef_mat.topRows(nrow_coef) = unvectorize(reg_record->coef_record.row(i).head(num_alpha).transpose(), dim);
		if (include_mean) {
			coef_mat.bottomRows(1) = reg_record->coef_record.row(i).tail(dim);
		}
		reg_record->updateDiag(i, sv_update, sv_sig); // D^1/2
		contem_mat = build_inv_lower(dim, reg_record->contem_coef_record.row(i)); // L
	}
	void updateVariance() override {
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
		lpl[h] += sv_update.sum() / 2 - dim * log(2 * M_PI) / 2 - ((-sv_update / 2).array().exp() * (contem_mat * (post_mean - valid_vec)).array()).matrix().squaredNorm() / 2;
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
class McmcVarForecaster : public BaseForecaster {
public:
	McmcVarForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true
	)
	: BaseForecaster(records, step, response_mat, lag, include_mean, filter_stable, seed, sv) {
		if (stable_filter) {
			reg_record->subsetStable(num_alpha, 1);
			num_sim = reg_record->coef_record.rows();
			if (num_sim == 0) {
				STOP("No stable MCMC draws");
			}
		}
	}
	virtual ~McmcVarForecaster() = default;

protected:
	using BaseForecaster::reg_record;
	using BaseForecaster::stable_filter;
	using BaseForecaster::num_alpha;
	using BaseForecaster::num_sim;
	using BaseForecaster::post_mean;
	using BaseForecaster::coef_mat;
	using BaseForecaster::last_pvec;
	void computeMean() override {
		post_mean = coef_mat.transpose() * last_pvec;
	}
};

/**
 * @brief Forecast class of Bayesian VHAR based on `McmcTriangular`
 * 
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 */
template <typename BaseForecaster = RegForecaster>
class McmcVharForecaster : public BaseForecaster {
public:
	McmcVharForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true
	)
	: BaseForecaster(records, step, response_mat, month, include_mean, filter_stable, seed, sv), har_trans(har_trans) {
		if (stable_filter) {
			reg_record->subsetStable(num_alpha, 1, har_trans.topLeftCorner(3 * dim, month * dim));
			num_sim = reg_record->coef_record.rows();
			if (num_sim == 0) {
				STOP("No stable MCMC draws");
			}
		}
	}
	virtual ~McmcVharForecaster() = default;
	
protected:
	using BaseForecaster::reg_record;
	using BaseForecaster::stable_filter;
	using BaseForecaster::dim;
	using BaseForecaster::num_alpha;
	using BaseForecaster::num_sim;
	using BaseForecaster::post_mean;
	using BaseForecaster::coef_mat;
	using BaseForecaster::last_pvec;
	Eigen::MatrixXd har_trans;
	void computeMean() override {
		post_mean = coef_mat.transpose() * har_trans * last_pvec;
	}
};

/**
 * @brief Bayesian VAR forecast class with sparse draw induced by posterior summary
 * 
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 */
template <typename BaseForecaster = RegForecaster>
class McmcVarSelectForecaster : public McmcVarForecaster<BaseForecaster> {
public:
	McmcVarSelectForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		double level, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true
	)
	: McmcVarForecaster<BaseForecaster>(records, step, response_mat, lag, include_mean, filter_stable, seed, sv),
		activity_graph(unvectorize(reg_record->computeActivity(level), dim)) {}
	McmcVarSelectForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		const Eigen::MatrixXd& selection, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true
	)
	: McmcVarForecaster<BaseForecaster>(records, step, response_mat, lag, include_mean, filter_stable, seed, sv),
		activity_graph(selection) {}
	
	virtual ~McmcVarSelectForecaster() = default;

protected:
	using McmcVarForecaster<BaseForecaster>::dim;
	using McmcVarForecaster<BaseForecaster>::reg_record;
	using McmcVarForecaster<BaseForecaster>::post_mean;
	using McmcVarForecaster<BaseForecaster>::coef_mat;
	using McmcVarForecaster<BaseForecaster>::last_pvec;
	void computeMean() override {
		post_mean = last_pvec.transpose() * (activity_graph.array() * coef_mat.array()).matrix();
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
class McmcVharSelectForecaster : public McmcVharForecaster<BaseForecaster> {
public:
	McmcVharSelectForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		double level, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true
	)
	: McmcVharForecaster<BaseForecaster>(records, step, response_mat, har_trans, month, include_mean, filter_stable, seed, sv),
		activity_graph(unvectorize(reg_record->computeActivity(level), dim)) {}
	McmcVharSelectForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		const Eigen::MatrixXd& selection, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true
	)
	: McmcVharForecaster<BaseForecaster>(records, step, response_mat, har_trans, month, include_mean, filter_stable, seed, sv),
		activity_graph(selection) {}
	
	virtual ~McmcVharSelectForecaster() = default;

protected:
	using McmcVharForecaster<BaseForecaster>::dim;
	using McmcVharForecaster<BaseForecaster>::reg_record;
	using McmcVharForecaster<BaseForecaster>::post_mean;
	using McmcVharForecaster<BaseForecaster>::coef_mat;
	using McmcVharForecaster<BaseForecaster>::last_pvec;
	using McmcVharForecaster<BaseForecaster>::har_trans;
	void computeMean() override {
		post_mean = last_pvec.transpose() * har_trans.transpose() * (activity_graph.array() * coef_mat.array()).matrix();
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
 * @param fit_record `LIST` of MCMC draws
 * @param seed_chain Random seed for each chain
 * @param include_mean If `true`, include constant term
 * @param stable If `true`, filter stable draws
 * @param nthreads Number of OpenMP threads
 * @param sv Use stochastic volaility when forecasting
 * @param har_trans VHAR transformation matrix
 * @return std::vector<std::unique_ptr<BaseForecaster>> Vector of forecast class smart pointer corresponding to each chain
 */
template <typename BaseForecaster = RegForecaster>
inline std::vector<std::unique_ptr<BaseForecaster>> initialize_forecaster(
	int num_chains, int ord, int step, const Eigen::MatrixXd& response_mat,
	bool sparse, double level, LIST& fit_record,
	Eigen::Ref<const Eigen::VectorXi> seed_chain, bool include_mean, bool stable, int nthreads,
	bool sv = true, Optional<Eigen::MatrixXd> har_trans = NULLOPT
) {
	bool activity = (level > 0); // Optional<double> level = NULLOPT
	if (sparse && activity) {
		STOP("If 'level > 0', 'spare' should be false.");
	}
	using Records = typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type;
	std::vector<std::unique_ptr<BaseForecaster>> forecaster_ptr(num_chains);
	STRING coef_name = har_trans ? (sparse ? "phi_sparse_record" : "phi_record") : (sparse ? "alpha_sparse_record" : "alpha_record");
	STRING a_name = sparse ? "a_sparse_record" : "a_record";
	STRING c_name = sparse ? "c_sparse_record" : "c_record";
	for (int i = 0; i < num_chains; ++i) {
		std::unique_ptr<Records> reg_record;
		initialize_record(reg_record, i, fit_record, include_mean, coef_name, a_name, c_name);
		if (har_trans && !activity) {
			forecaster_ptr[i] = std::make_unique<McmcVharForecaster<BaseForecaster>>(
				*reg_record, step, response_mat,
				*har_trans, ord,
				include_mean, stable, static_cast<unsigned int>(seed_chain[i]),
				sv
			);
		} else if (!har_trans && !activity) {
			forecaster_ptr[i] = std::make_unique<McmcVarForecaster<BaseForecaster>>(
				*reg_record, step, response_mat,
				ord,
				include_mean, stable, static_cast<unsigned int>(seed_chain[i]),
				sv
			);
		} else if (har_trans && activity) {
			forecaster_ptr[i] = std::make_unique<McmcVharSelectForecaster<BaseForecaster>>(
				*reg_record, level, step, response_mat,
				*har_trans, ord,
				include_mean, stable, static_cast<unsigned int>(seed_chain[i]),
				sv
			);
		} else {
			forecaster_ptr[i] = std::make_unique<McmcVarSelectForecaster<BaseForecaster>>(
				*reg_record, level, step, response_mat,
				ord,
				include_mean, stable, static_cast<unsigned int>(seed_chain[i]),
				sv
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
class McmcForecastRun {
public:
	McmcForecastRun(
		int num_chains, int lag, int step, const Eigen::MatrixXd& response_mat,
		bool sparse, double level, LIST& fit_record,
		const Eigen::VectorXi& seed_chain, bool include_mean, bool stable, int nthreads,
		bool sv = true
	)
	: num_chains(num_chains), nthreads(nthreads), density_forecast(num_chains), forecaster(num_chains) {
		forecaster = initialize_forecaster<BaseForecaster>(
			num_chains, lag, step, response_mat, sparse, level,
			fit_record, seed_chain, include_mean,
			stable, nthreads, sv
		);
	}
	McmcForecastRun(
		int num_chains, int week, int month, int step, const Eigen::MatrixXd& response_mat,
		bool sparse, double level, LIST& fit_record,
		const Eigen::VectorXi& seed_chain, bool include_mean, bool stable, int nthreads,
		bool sv = true
	)
	: num_chains(num_chains), nthreads(nthreads), density_forecast(num_chains) {
		Eigen::MatrixXd har_trans = build_vhar(response_mat.cols(), week, month, include_mean);
		forecaster = initialize_forecaster<BaseForecaster>(
			num_chains, month, step, response_mat, sparse, level,
			fit_record, seed_chain, include_mean,
			stable, nthreads, sv, har_trans
		);
	}
	McmcForecastRun(
		int num_chains, int month, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans,
		bool sparse, double level, LIST& fit_record,
		const Eigen::VectorXi& seed_chain, bool include_mean, bool stable, int nthreads,
		bool sv = true
	)
	: num_chains(num_chains), nthreads(nthreads), density_forecast(num_chains) {
		forecaster = initialize_forecaster<BaseForecaster>(
			num_chains, month, step, response_mat, sparse, level,
			fit_record, seed_chain, include_mean,
			stable, nthreads, sv, har_trans
		);
	}
	virtual ~McmcForecastRun() = default;

	/**
	 * @brief Forecast
	 * 
	 */
	void forecast() {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; ++chain) {
			density_forecast[chain] = forecaster[chain]->forecastDensity();
			forecaster[chain].reset(); // free the memory by making nullptr
		}
	}

	/**
	 * @brief Return forecast draws
	 * 
	 * @return std::vector<Eigen::MatrixXd> Forecast density of each chain
	 */
	std::vector<Eigen::MatrixXd> returnForecast() {
		forecast();
		return density_forecast;
	}

private:
	int num_chains;
	int nthreads;
	std::vector<Eigen::MatrixXd> density_forecast;
	std::vector<std::unique_ptr<BaseForecaster>> forecaster;
};

/**
 * @brief Interface class for Out-of-sample forecasting with MCMC
 * 
 * @tparam BaseForecaster 
 */
template <typename BaseForecaster = RegForecaster>
class McmcOutforecastInterface {
public:
	virtual ~McmcOutforecastInterface() = default;

	/**
	 * @brief Out-of-sample forecasting
	 * 
	 */
	virtual void forecast() = 0;

	/**
	 * @brief Return out-of-sample forecasting draws
	 * 
	 * @return LIST `LIST` containing forecast draws. Include ALPL when `get_lpl` is `true`.
	 */
	virtual LIST returnForecast() = 0;
};

/**
 * @brief Out-of-sample forecasting class
 * 
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 * @tparam isUpdate MCMC again in the new window
 */
template <typename BaseForecaster = RegForecaster, bool isUpdate = true>
class McmcOutforecastRun : public McmcOutforecastInterface<BaseForecaster> {
public:
	McmcOutforecastRun(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads, bool sv = true
	)
	: num_window(y.rows()), dim(y.cols()), num_test(y_test.rows()), num_horizon(num_test - step + 1), step(step),
		lag(lag), num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin), nthreads(nthreads),
		include_mean(include_mean), stable_filter(stable), sparse(sparse), get_lpl(get_lpl),
		sv(sv), display_progress(display_progress), level(level), seed_forecast(seed_forecast),
		roll_mat(num_horizon), roll_y0(num_horizon), y_test(y_test),
		model(num_horizon), forecaster(num_horizon),
		out_forecast(num_horizon, std::vector<Eigen::MatrixXd>(num_chains)),
		lpl_record(Eigen::MatrixXd::Zero(num_horizon, num_chains)) {
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
		if (level > 0) {
			sparse = false;
		}
	}
	virtual ~McmcOutforecastRun() = default;

	void forecast() override {
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

	LIST returnForecast() override {
		forecast();
		LIST res = CREATE_LIST(NAMED("forecast") = WRAP(out_forecast));
		if (get_lpl) {
			// res["lpl"] = CAST_DOUBLE(lpl_record.mean());
			// res["lpl"] = CAST_VECTOR(lpl_record.rowwise().mean());
			res["lpl"] = lpl_record;
		}
		return res;
	}

protected:
	using BaseMcmc = typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, McmcReg, McmcSv>::type;
	using RecordType = typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type;
	int num_window, dim, num_test, num_horizon, step;
	int lag, num_chains, num_iter, num_burn, thin, nthreads;
	bool include_mean, stable_filter, sparse, get_lpl, sv, display_progress;
	double level;
	Eigen::VectorXi seed_forecast;
	std::vector<Eigen::MatrixXd> roll_mat;
	std::vector<Eigen::MatrixXd> roll_y0;
	Eigen::MatrixXd y_test;
	std::vector<std::vector<std::unique_ptr<BaseMcmc>>> model;
	std::vector<std::vector<std::unique_ptr<BaseForecaster>>> forecaster;
	std::vector<std::vector<Eigen::MatrixXd>> out_forecast;
	Eigen::MatrixXd lpl_record;

	/**
	 * @brief Define input in each window
	 * 
	 * @param y Entire data including validation set
	 */
	virtual void initData(const Eigen::MatrixXd& y) = 0;

	/**
	 * @brief Initialize forecaster
	 * 
	 * @param fit_record MCMC draw `LIST`
	 */
	virtual void initForecaster(LIST& fit_record) = 0;

	/**
	 * @brief Initialize CTA
	 * 
	 * @param param_reg `LIST` of CTA hyperparameters
	 * @param param_prior `LIST` of shrinkage prior hyperparameters
	 * @param param_intercept `LIST` of Normal prior hyperparameters for constant term
	 * @param param_init `LIST_OF_LIST` for initial values
	 * @param prior_type Shrinkage prior number
	 * @param grp_id Minnesota group unique id
	 * @param own_id own-lag id
	 * @param cross_id cross-lag id
	 * @param grp_mat Minnesota group matrix
	 * @param seed_chain Random seed for each chain
	 */
	virtual void initMcmc(
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		const Eigen::MatrixXi& seed_chain
	) = 0;

	/**
	 * @brief Define VAR or VHAR design matrix
	 * 
	 * @param window Window index
	 * @return Eigen::MatrixXd Design matrix
	 */
	virtual Eigen::MatrixXd buildDesign(int window) = 0;

	/**
	 * @brief Replace the forecast smart pointer given MCMC result
	 * 
	 * @param reg_record MCMC record
	 * @param window Window index
	 * @param chain Chain index
	 */
	virtual void updateForecaster(RecordType& reg_record, int window, int chain) = 0;

	/**
	 * @brief Initialize every member of `McmcOutforecastRun`
	 * 
	 * @param y Response matrix
	 * @param fit_record `LIST` of MCMC draws
	 * @param param_reg `LIST` of CTA hyperparameters
	 * @param param_prior `LIST` of shrinkage prior hyperparameters
	 * @param param_intercept `LIST` of Normal prior hyperparameters for constant term
	 * @param param_init `LIST_OF_LIST` for initial values
	 * @param prior_type Shrinkage prior number
	 * @param grp_id Minnesota group unique id
	 * @param own_id own-lag id
	 * @param cross_id cross-lag id
	 * @param grp_mat Minnesota group matrix
	 * @param seed_chain Random seed for each chain
	 */
	void initialize(
		const Eigen::MatrixXd& y, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		const Eigen::MatrixXi& seed_chain
	) {
		initData(y);
		initForecaster(fit_record);
		using is_mcmc = std::integral_constant<bool, isUpdate>;
		if (is_mcmc::value) {
			initMcmc(
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat, seed_chain
			);
		}
	}

	/**
	 * @brief Conduct MCMC and update forecast pointer
	 * 
	 * @param window Window index
	 * @param chain Chain index
	 */
	void runGibbs(int window, int chain) {
		std::string log_name = fmt::format("Chain {} / Window {}", chain + 1, window + 1);
		auto logger = spdlog::get(log_name);
		if (logger == nullptr) {
			logger = SPDLOG_SINK_MT(log_name);
		}
		logger->set_pattern("[%n] [Thread " + std::to_string(omp_get_thread_num()) + "] %v");
		int logging_freq = num_iter / 20; // 5 percent
		if (logging_freq == 0) {
			logging_freq = 1;
		}
		bvharinterrupt();
		for (int i = 0; i < num_burn; ++i) {
			model[window][chain]->doWarmUp();
			if (display_progress && (i + 1) % logging_freq == 0) {
				logger->info("{} / {} (Warmup)", i + 1, num_iter);
			}
		}
		logger->flush();
		for (int i = num_burn; i < num_iter; ++i) {
			if (bvharinterrupt::is_interrupted()) {
				RecordType reg_record = model[window][chain]->template returnStructRecords<RecordType>(0, thin, sparse);
				logger->warn("User interrupt in {} / {}", i + 1, num_iter);
				break;
			}
			model[window][chain]->doPosteriorDraws();
			if (display_progress && (i + 1) % logging_freq == 0) {
				logger->info("{} / {} (Sampling)", i + 1, num_iter);
			}
		}
		RecordType reg_record = model[window][chain]->template returnStructRecords<RecordType>(0, thin, sparse);
		updateForecaster(reg_record, window, chain);
		model[window][chain].reset();
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
		using is_mcmc = std::integral_constant<bool, isUpdate>;
		if (window != 0 && is_mcmc::value) {
			runGibbs(window, chain);
		}
		Eigen::VectorXd valid_vec = y_test.row(step);
		out_forecast[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
		lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
		forecaster[window][chain].reset(); // free the memory by making nullpt
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
class McmcRollforecastRun : public McmcOutforecastRun<BaseForecaster, isUpdate> {
public:
	McmcRollforecastRun(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads, bool sv = true
	)
	: McmcOutforecastRun<BaseForecaster, isUpdate>(
			y, lag, num_chains, num_iter, num_burn, thin, sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test, get_lpl,
			seed_chain, seed_forecast, display_progress, nthreads, sv
		) {}
	virtual ~McmcRollforecastRun() = default;

protected:
	using typename McmcOutforecastRun<BaseForecaster, isUpdate>::BaseMcmc;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::num_window;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::dim;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::num_test;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::num_horizon;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::lag;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::num_chains;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::num_iter;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::num_burn;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::include_mean;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::roll_mat;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::roll_y0;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::y_test;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::model;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::buildDesign;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::initialize;
	void initData(const Eigen::MatrixXd& y) override {
		Eigen::MatrixXd tot_mat(num_window + num_test, dim);
		tot_mat << y,
							 y_test;
		for (int i = 0; i < num_horizon; ++i) {
			roll_mat[i] = tot_mat.middleRows(i, num_window);
			roll_y0[i] = build_y0(roll_mat[i], lag, lag + 1);
		}
	}
	void initMcmc(
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		const Eigen::MatrixXi& seed_chain
	) override {
		for (int window = 0; window < num_horizon; ++window) {
			Eigen::MatrixXd design = buildDesign(window);
			model[window] = initialize_mcmc<BaseMcmc, isGroup>(
				num_chains, num_iter - num_burn, design, roll_y0[window],
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain.row(window)
			);
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
class McmcExpandforecastRun : public McmcOutforecastRun<BaseForecaster, isUpdate> {
public:
	McmcExpandforecastRun(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads, bool sv = true
	)
	: McmcOutforecastRun<BaseForecaster, isUpdate>(
			y, lag, num_chains, num_iter, num_burn, thin, sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test, get_lpl,
			seed_chain, seed_forecast, display_progress, nthreads, sv
		) {}
	virtual ~McmcExpandforecastRun() = default;

protected:
	using typename McmcOutforecastRun<BaseForecaster, isUpdate>::BaseMcmc;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::num_window;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::dim;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::num_test;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::num_horizon;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::lag;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::num_chains;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::num_iter;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::num_burn;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::include_mean;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::roll_mat;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::roll_y0;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::y_test;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::model;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::buildDesign;
	using McmcOutforecastRun<BaseForecaster, isUpdate>::initialize;
	void initData(const Eigen::MatrixXd& y) override {
		Eigen::MatrixXd tot_mat(num_window + num_test, dim);
		tot_mat << y,
							 y_test;
		for (int i = 0; i < num_horizon; ++i) {
			roll_mat[i] = tot_mat.topRows(num_window + i);
			roll_y0[i] = build_y0(roll_mat[i], lag, lag + 1);
		}
	}
	void initMcmc(
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		const Eigen::MatrixXi& seed_chain
	) override {
		for (int window = 0; window < num_horizon; ++window) {
			Eigen::MatrixXd design = buildDesign(window);
			if (CONTAINS(param_reg, "initial_mean")) {
				// BaseMcmc == McmcSv
				model[window] = initialize_mcmc<BaseMcmc, isGroup>(
					num_chains, num_iter - num_burn, design, roll_y0[window],
					param_reg, param_prior, param_intercept, param_init, prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain.row(window),
					roll_y0[window].rows()
				);
			} else {
				// BaseMcmc == McmcReg
				model[window] = initialize_mcmc<BaseMcmc, isGroup>(
					num_chains, num_iter - num_burn, design, roll_y0[window],
					param_reg, param_prior, param_intercept, param_init, prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain.row(window)
				);
			}
			roll_mat[window].resize(0, 0); // free the memory
		}
	}
};

/**
 * @brief Out-of-sample forecast class for Bayesian VAR
 * 
 * @tparam BaseOutForecast `McmcRollforecastRun` or `McmcExpandforecastRun`
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 * @tparam isGroup If `true`, use group shrinkage parameter
 * @tparam isUpdate MCMC again in the new window
 */
template <template <typename, bool, bool> class BaseOutForecast = McmcRollforecastRun, typename BaseForecaster = RegForecaster, bool isGroup = true, bool isUpdate = true>
class McmcVarforecastRun : public BaseOutForecast<BaseForecaster, isGroup, isUpdate> {
public:
	McmcVarforecastRun(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads, bool sv = true
	)
	: BaseOutForecast<BaseForecaster, isGroup, isUpdate>(
			y, lag, num_chains, num_iter, num_burn, thin, sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test, get_lpl,
			seed_chain, seed_forecast, display_progress, nthreads, sv
		) {
		initialize(
			y, fit_record, param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, seed_chain
		);
	}
	virtual ~McmcVarforecastRun() = default;

protected:
	using typename BaseOutForecast<BaseForecaster, isGroup, isUpdate>::BaseMcmc;
	using typename BaseOutForecast<BaseForecaster, isGroup, isUpdate>::RecordType;
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
	void initForecaster(LIST& fit_record) override {
		using is_mcmc = std::integral_constant<bool, isUpdate>;
		if (is_mcmc::value) {
			forecaster[0] = initialize_forecaster<BaseForecaster>(
				num_chains, lag, step, roll_y0[0], sparse, level,
				fit_record, seed_forecast, include_mean,
				stable_filter, nthreads, sv
			);
		} else {
			for (int i = 0; i < num_horizon; ++i) {
				forecaster[i] = initialize_forecaster<BaseForecaster>(
					num_chains, lag, step, roll_y0[i], sparse, level,
					fit_record, seed_forecast, include_mean,
					stable_filter, nthreads, sv
				);
			}
		}
	}
	Eigen::MatrixXd buildDesign(int window) override {
		return build_x0(roll_mat[window], lag, include_mean);
	}
	void updateForecaster(RecordType& reg_record, int window, int chain) override {
		if (level > 0) {
			forecaster[window][chain] = std::make_unique<McmcVarSelectForecaster<BaseForecaster>>(
				reg_record, level, step, roll_y0[window], lag, include_mean,
				stable_filter, static_cast<unsigned int>(seed_forecast[chain]), sv
			);
		} else {
			forecaster[window][chain] = std::make_unique<McmcVarForecaster<BaseForecaster>>(
				reg_record, step, roll_y0[window], lag, include_mean,
				stable_filter, static_cast<unsigned int>(seed_forecast[chain]), sv
			);
		}
	}
};

/**
 * @brief Out-of-sample forecast class for Bayesian VHAR
 * 
 * @tparam BaseOutForecast `McmcRollforecastRun` or `McmcExpandforecastRun`
 * @tparam BaseForecaster `RegForecaster` or `SvForecaster`
 * @tparam isGroup If `true`, use group shrinkage parameter
 * @tparam isUpdate MCMC again in the new window
 */
template <template <typename, bool, bool> class BaseOutForecast = McmcRollforecastRun, typename BaseForecaster = RegForecaster, bool isGroup = true, bool isUpdate = true>
class McmcVharforecastRun : public BaseOutForecast<BaseForecaster, isGroup, isUpdate> {
public:
	McmcVharforecastRun(
		const Eigen::MatrixXd& y, int week, int month, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads, bool sv = true
	)
	: BaseOutForecast<BaseForecaster, isGroup, isUpdate>(
			y, month, num_chains, num_iter, num_burn, thin, sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test, get_lpl,
			seed_chain, seed_forecast, display_progress, nthreads, sv
		),
		har_trans(build_vhar(dim, week, month, include_mean)) {
		initialize(
			y, fit_record, param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, seed_chain
		);
	}
	virtual ~McmcVharforecastRun() = default;

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
	Eigen::MatrixXd har_trans;
	void initForecaster(LIST& fit_record) override {
		using is_mcmc = std::integral_constant<bool, isUpdate>;
		if (is_mcmc::value) {
			forecaster[0] = initialize_forecaster<BaseForecaster>(
				num_chains, lag, step, roll_y0[0], sparse, level,
				fit_record, seed_forecast, include_mean,
				stable_filter, nthreads, sv,
				har_trans
			);
		} else {
			for (int i = 0; i < num_horizon; ++i) {
				forecaster[i] = initialize_forecaster<BaseForecaster>(
					num_chains, lag, step, roll_y0[i], sparse, level,
					fit_record, seed_forecast, include_mean,
					stable_filter, nthreads, sv,
					har_trans
				);
			}
		}
	}
	Eigen::MatrixXd buildDesign(int window) override {
		return build_x0(roll_mat[window], lag, include_mean) * har_trans.transpose();
	}
	void updateForecaster(RecordType& reg_record, int window, int chain) override {
		if (level > 0) {
			forecaster[window][chain] = std::make_unique<McmcVharSelectForecaster<BaseForecaster>>(
				reg_record, level, step, roll_y0[window], har_trans, lag, include_mean,
				stable_filter, static_cast<unsigned int>(seed_forecast[chain]), sv
			);
		} else {
			forecaster[window][chain] = std::make_unique<McmcVharForecaster<BaseForecaster>>(
				reg_record, step, roll_y0[window], har_trans, lag, include_mean,
				stable_filter, static_cast<unsigned int>(seed_forecast[chain]), sv
			);
		}
	}
};

template <template <typename, bool, bool> class BaseOutForecast = McmcRollforecastRun, typename BaseForecaster = RegForecaster>
inline std::unique_ptr<McmcOutforecastInterface<BaseForecaster>> initialize_outforecaster(
	const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
	bool sparse, double level, LIST& fit_record, bool run_mcmc,
	LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type, bool ggl,
	const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
	bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test,
	bool get_lpl, const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
	const bool sv
) {
	if (ggl && run_mcmc) {
		return std::make_unique<McmcVarforecastRun<BaseOutForecast, BaseForecaster, true, true>>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	} else if (ggl && !run_mcmc) {
		return std::make_unique<McmcVarforecastRun<BaseOutForecast, BaseForecaster, true, false>>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	} else if (!ggl && run_mcmc) {
		return std::make_unique<McmcVarforecastRun<BaseOutForecast, BaseForecaster, false, true>>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	}
	return std::make_unique<McmcVarforecastRun<BaseOutForecast, BaseForecaster, false, false>>(
		y, lag, num_chains, num_iter, num_burn, thinning,
		sparse, level, fit_record,
		param_reg, param_prior, param_intercept, param_init, prior_type,
		grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
		get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
	);
}

template <template <typename, bool, bool> class BaseOutForecast = McmcRollforecastRun, typename BaseForecaster = RegForecaster>
inline std::unique_ptr<McmcOutforecastInterface<BaseForecaster>> initialize_outforecaster(
	const Eigen::MatrixXd& y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
	bool sparse, double level, LIST& fit_record, bool run_mcmc,
	LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type, bool ggl,
	const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
	bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test,
	bool get_lpl, const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
	const bool sv
) {
	if (ggl && run_mcmc) {
		return std::make_unique<McmcVharforecastRun<BaseOutForecast, BaseForecaster, true, true>>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	} else if (ggl && !run_mcmc) {
		return std::make_unique<McmcVharforecastRun<BaseOutForecast, BaseForecaster, true, false>>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	} else if (!ggl && run_mcmc) {
		return std::make_unique<McmcVharforecastRun<BaseOutForecast, BaseForecaster, false, true>>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	}
	return std::make_unique<McmcVharforecastRun<BaseOutForecast, BaseForecaster, false, false>>(
		y, week, month, num_chains, num_iter, num_burn, thinning,
		sparse, level, fit_record,
		param_reg, param_prior, param_intercept, param_init, prior_type,
		grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
		get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
	);
}

} // namespace bvhar

#endif // BVHAR_BAYES_TRIANGULAR_FORECASTER_H