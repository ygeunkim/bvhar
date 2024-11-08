#ifndef BVHARFORECASTER_H
#define BVHARFORECASTER_H

#include "bvharmcmc.h"

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
template <typename BaseForecaster> class McmcOutforecastRun;
template <typename BaseForecaster> class McmcRollforecastRun;
template <typename BaseForecaster> class McmcExpandforecastRun;
template <template <typename> class BaseOutForecast, typename BaseForecaster> class McmcVarforecastRun;
template <template <typename> class BaseOutForecast, typename BaseForecaster> class McmcVharforecastRun;

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
	Eigen::VectorXd returnLplRecord() {
		return lpl;
	}
	double returnLpl() {
		return lpl.mean();
	}

protected:
	std::unique_ptr<RegRecords> reg_record;
	boost::random::mt19937 rng;
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
	virtual void computeMean() = 0;
	virtual void updateParams(int i) = 0;
	virtual void updateVariance() = 0;
	virtual void updateLpl(int h, const Eigen::VectorXd& valid_vec) = 0;
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

class RegForecaster : public McmcForecaster {
public:
	RegForecaster(const LdltRecords& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true)
	: McmcForecaster(records, step, response_mat, ord, include_mean, filter_stable, seed, sv) {
		reg_record = std::make_unique<LdltRecords>(records);
	}
	virtual ~RegForecaster() = default;

protected:
	// virtual void computeMean() = 0;
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
		lpl[h] += sv_update.array().log().sum() - dim * log(2 * M_PI) / 2 - (sv_update.cwiseInverse().array() * (contem_mat * (post_mean - valid_vec)).array()).matrix().squaredNorm() / 2;
	}
};

class SvForecaster : public McmcForecaster {
public:
	SvForecaster(const SvRecords& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean, bool filter_stable, unsigned int seed, bool sv)
	: McmcForecaster(records, step, response_mat, ord, include_mean, filter_stable, seed, sv),
		sv(sv), sv_sig(Eigen::VectorXd::Zero(dim)) {
		reg_record = std::make_unique<SvRecords>(records);
	}
	virtual ~SvForecaster() = default;

protected:
	virtual void computeMean() = 0;
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

template <typename BaseForecaster = RegForecaster>
class McmcVarForecaster : public BaseForecaster {
public:
	McmcVarForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true
	)
	: BaseForecaster(records, step, response_mat, lag, include_mean, filter_stable, seed, sv) {
		if (stable_filter) {
			reg_record->subsetStable(num_alpha, 1.05);
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

template <typename BaseForecaster = RegForecaster>
class McmcVharForecaster : public BaseForecaster {
public:
	McmcVharForecaster(
		const typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type& records,
		int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, unsigned int seed, bool sv = true
	)
	: BaseForecaster(records, step, response_mat, month, include_mean, filter_stable, seed, sv), har_trans(har_trans.sparseView()) {
		if (stable_filter) {
			reg_record->subsetStable(num_alpha, 1.05, har_trans.topLeftCorner(3 * dim, month * dim).sparseView());
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
	Eigen::SparseMatrix<double> har_trans;
	void computeMean() override {
		post_mean = coef_mat.transpose() * har_trans * last_pvec;
	}
};

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

inline void initialize_record(std::unique_ptr<LdltRecords>& record, int chain_id, LIST& fit_record, bool include_mean, STRING& coef_name, STRING& a_name, STRING& c_name) {
	PY_LIST coef_list = fit_record[coef_name];
	PY_LIST a_list = fit_record[a_name];
	PY_LIST d_list = fit_record["d_record"];
	if (include_mean) {
		PY_LIST c_list = fit_record[c_name];
		record = std::make_unique<LdltRecords>(
			CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			CAST<Eigen::MatrixXd>(c_list[chain_id]),
			CAST<Eigen::MatrixXd>(a_list[chain_id]),
			CAST<Eigen::MatrixXd>(d_list[chain_id])
		);
	} else {
		record = std::make_unique<LdltRecords>(
			CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			CAST<Eigen::MatrixXd>(a_list[chain_id]),
			CAST<Eigen::MatrixXd>(d_list[chain_id])
		);
	}
}

inline void initialize_record(std::unique_ptr<SvRecords>& record, int chain_id, LIST& fit_record, bool include_mean, STRING& coef_name, STRING& a_name, STRING& c_name) {
	PY_LIST coef_list = fit_record[coef_name];
	PY_LIST a_list = fit_record[a_name];
	PY_LIST h_list = fit_record["h_record"];
	PY_LIST sigh_list = fit_record["sigh_record"];
	if (include_mean) {
		PY_LIST c_list = fit_record[c_name];
		record = std::make_unique<SvRecords>(
			CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			CAST<Eigen::MatrixXd>(c_list[chain_id]),
			CAST<Eigen::MatrixXd>(h_list[chain_id]),
			CAST<Eigen::MatrixXd>(a_list[chain_id]),
			CAST<Eigen::MatrixXd>(sigh_list[chain_id])
		);
	} else {
		record = std::make_unique<SvRecords>(
			CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			CAST<Eigen::MatrixXd>(h_list[chain_id]),
			CAST<Eigen::MatrixXd>(a_list[chain_id]),
			CAST<Eigen::MatrixXd>(sigh_list[chain_id])
		);
	}
}

template <typename BaseForecaster = RegForecaster>
inline std::vector<std::unique_ptr<BaseForecaster>> initialize_forecaster(
	int num_chains, int ord, int step, const Eigen::MatrixXd& response_mat,
	bool sparse, double level, LIST& fit_record,
	Eigen::Ref<const Eigen::VectorXi> seed_chain, bool include_mean, bool stable, int nthreads,
	bool sv = true, Optional<Eigen::MatrixXd> har_trans = NULLOPT
) {
	bool activity = (level > 0); // Optional<double> level = NULLOPT
	if (!sparse && activity) {
		STOP("If 'level > 0', 'spare' should be true."); // change this later: sparse && activity
	}
	using Records = typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type;
	std::vector<std::unique_ptr<BaseForecaster>> forecaster_ptr(num_chains);
	STRING coef_name = har_trans ? (sparse ? "phi_sparse_record" : "phi_record") : (sparse ? "alpha_sparse_record" : "alpha_record");
	STRING a_name = sparse ? "a_sparse_record" : "a_record";
	STRING c_name = sparse ? "c_sparse_record" : "c_record";
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_chains; ++i) {
		std::unique_ptr<Records> reg_record;
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
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
	}
	return forecaster_ptr;
}

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
	std::vector<Eigen::MatrixXd> density_forecast;
	std::vector<std::unique_ptr<BaseForecaster>> forecaster;
};

template <typename BaseForecaster = RegForecaster>
class McmcOutforecastRun {
public:
	McmcOutforecastRun(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, int nthreads, bool sv = true
	)
	: num_window(y.rows()), dim(y.cols()), num_test(y_test.rows()), num_horizon(num_test - step + 1), step(step),
		lag(lag), num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin), nthreads(nthreads),
		include_mean(include_mean), stable_filter(stable), sparse(sparse), get_lpl(get_lpl),
		sv(sv), level(level), seed_forecast(seed_forecast),
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
	}
	virtual ~McmcOutforecastRun() = default;
	LIST returnForecast() {
		forecast();
		LIST res = CREATE_LIST(NAMED("forecast") = WRAP(out_forecast));
		if (get_lpl) {
			res["lpl"] = CAST_DOUBLE(lpl_record.mean());
		}
		return res;
	}

protected:
	using BaseMcmc = typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, McmcReg, McmcSv>::type;
	using RecordType = typename std::conditional<std::is_same<BaseForecaster, RegForecaster>::value, LdltRecords, SvRecords>::type;
	int num_window, dim, num_test, num_horizon, step;
	int lag, num_chains, num_iter, num_burn, thin, nthreads;
	bool include_mean, stable_filter, sparse, get_lpl, sv;
	double level;
	Eigen::VectorXi seed_forecast;
	std::vector<Eigen::MatrixXd> roll_mat;
	std::vector<Eigen::MatrixXd> roll_y0;
	Eigen::MatrixXd y_test;
	std::vector<std::vector<std::unique_ptr<BaseMcmc>>> model;
	std::vector<std::vector<std::unique_ptr<BaseForecaster>>> forecaster;
	std::vector<std::vector<Eigen::MatrixXd>> out_forecast;
	Eigen::MatrixXd lpl_record;
	virtual void initData(const Eigen::MatrixXd& y) = 0;
	virtual void initForecaster(LIST& fit_record) = 0;
	virtual void initMcmc(
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		const Eigen::MatrixXi& seed_chain
	) = 0;
	virtual Eigen::MatrixXd buildDesign(int window) = 0;
	virtual void updateForecaster(RecordType& reg_record, int window, int chain) = 0;
	void initialize(
		const Eigen::MatrixXd& y, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		const Eigen::MatrixXi& seed_chain
	) {
		initData(y);
		initForecaster(fit_record);
		initMcmc(
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, seed_chain
		);
	}
	void runGibbs(int window, int chain) {
		bvharinterrupt();
		for (int i = 0; i < num_iter; ++i) {
			if (bvharinterrupt::is_interrupted()) {
				RecordType reg_record = model[window][chain]->template returnStructRecords<RecordType>(num_burn, thin, sparse);
				break;
			}
			model[window][chain]->doPosteriorDraws();
		}
		RecordType reg_record = model[window][chain]->template returnStructRecords<RecordType>(num_burn, thin, sparse);
		updateForecaster(reg_record, window, chain);
	}
	void forecastWindow(int window, int chain) {
		if (window != 0) {
			runGibbs(window, chain);
		}
		Eigen::VectorXd valid_vec = y_test.row(step);
		out_forecast[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
		lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
		forecaster[window][chain].reset(); // free the memory by making nullpt
	}
	void forecast() {
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
};

template <typename BaseForecaster = RegForecaster>
class McmcRollforecastRun : public McmcOutforecastRun<BaseForecaster> {
public:
	McmcRollforecastRun(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, int nthreads, bool sv = true
	)
	: McmcOutforecastRun<BaseForecaster>(
			y, lag, num_chains, num_iter, num_burn, thin, sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test, get_lpl,
			seed_chain, seed_forecast, nthreads, sv
		) {}
	virtual ~McmcRollforecastRun() = default;

protected:
	using typename McmcOutforecastRun<BaseForecaster>::BaseMcmc;
	using McmcOutforecastRun<BaseForecaster>::num_window;
	using McmcOutforecastRun<BaseForecaster>::dim;
	using McmcOutforecastRun<BaseForecaster>::num_test;
	using McmcOutforecastRun<BaseForecaster>::num_horizon;
	using McmcOutforecastRun<BaseForecaster>::lag;
	using McmcOutforecastRun<BaseForecaster>::num_chains;
	using McmcOutforecastRun<BaseForecaster>::num_iter;
	using McmcOutforecastRun<BaseForecaster>::include_mean;
	using McmcOutforecastRun<BaseForecaster>::roll_mat;
	using McmcOutforecastRun<BaseForecaster>::roll_y0;
	using McmcOutforecastRun<BaseForecaster>::y_test;
	using McmcOutforecastRun<BaseForecaster>::model;
	using McmcOutforecastRun<BaseForecaster>::buildDesign;
	using McmcOutforecastRun<BaseForecaster>::initialize;
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
			model[window] = initialize_mcmc<BaseMcmc>(
				num_chains, num_iter, design, roll_y0[window],
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain.row(window)
			);
			roll_mat[window].resize(0, 0); // free the memory
		}
	}
};

template <typename BaseForecaster = RegForecaster>
class McmcExpandforecastRun : public McmcOutforecastRun<BaseForecaster> {
public:
	McmcExpandforecastRun(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, int nthreads, bool sv = true
	)
	: McmcOutforecastRun<BaseForecaster>(
			y, lag, num_chains, num_iter, num_burn, thin, sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test, get_lpl,
			seed_chain, seed_forecast, nthreads, sv
		) {}
	virtual ~McmcExpandforecastRun() = default;

protected:
	using typename McmcOutforecastRun<BaseForecaster>::BaseMcmc;
	using McmcOutforecastRun<BaseForecaster>::num_window;
	using McmcOutforecastRun<BaseForecaster>::dim;
	using McmcOutforecastRun<BaseForecaster>::num_test;
	using McmcOutforecastRun<BaseForecaster>::num_horizon;
	using McmcOutforecastRun<BaseForecaster>::lag;
	using McmcOutforecastRun<BaseForecaster>::num_chains;
	using McmcOutforecastRun<BaseForecaster>::num_iter;
	using McmcOutforecastRun<BaseForecaster>::include_mean;
	using McmcOutforecastRun<BaseForecaster>::roll_mat;
	using McmcOutforecastRun<BaseForecaster>::roll_y0;
	using McmcOutforecastRun<BaseForecaster>::y_test;
	using McmcOutforecastRun<BaseForecaster>::model;
	using McmcOutforecastRun<BaseForecaster>::buildDesign;
	using McmcOutforecastRun<BaseForecaster>::initialize;
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
				model[window] = initialize_mcmc<BaseMcmc>(
					num_chains, num_iter, design, roll_y0[window],
					param_reg, param_prior, param_intercept, param_init, prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain.row(window),
					roll_y0[window].rows()
				);
			} else {
				// BaseMcmc == McmcReg
				model[window] = initialize_mcmc<BaseMcmc>(
					num_chains, num_iter, design, roll_y0[window],
					param_reg, param_prior, param_intercept, param_init, prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain.row(window)
				);
			}
			roll_mat[window].resize(0, 0); // free the memory
		}
	}
};

template <template <typename> class BaseOutForecast = McmcRollforecastRun, typename BaseForecaster = RegForecaster>
class McmcVarforecastRun : public BaseOutForecast<BaseForecaster> {
public:
	McmcVarforecastRun(
		const Eigen::MatrixXd& y, int lag, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, int nthreads, bool sv = true
	)
	: BaseOutForecast<BaseForecaster>(
			y, lag, num_chains, num_iter, num_burn, thin, sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test, get_lpl,
			seed_chain, seed_forecast, nthreads, sv
		) {
		initialize(
			y, fit_record, param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, seed_chain
		);
	}
	virtual ~McmcVarforecastRun() = default;

protected:
	using typename BaseOutForecast<BaseForecaster>::BaseMcmc;
	using typename BaseOutForecast<BaseForecaster>::RecordType;
	using BaseOutForecast<BaseForecaster>::num_horizon;
	using BaseOutForecast<BaseForecaster>::step;
	using BaseOutForecast<BaseForecaster>::lag;
	using BaseOutForecast<BaseForecaster>::num_chains;
	using BaseOutForecast<BaseForecaster>::num_iter;
	using BaseOutForecast<BaseForecaster>::num_burn;
	using BaseOutForecast<BaseForecaster>::thin;
	using BaseOutForecast<BaseForecaster>::nthreads;
	using BaseOutForecast<BaseForecaster>::include_mean;
	using BaseOutForecast<BaseForecaster>::stable_filter;
	using BaseOutForecast<BaseForecaster>::sparse;
	using BaseOutForecast<BaseForecaster>::sv;
	using BaseOutForecast<BaseForecaster>::level;
	using BaseOutForecast<BaseForecaster>::seed_forecast;
	using BaseOutForecast<BaseForecaster>::roll_mat;
	using BaseOutForecast<BaseForecaster>::roll_y0;
	using BaseOutForecast<BaseForecaster>::model;
	using BaseOutForecast<BaseForecaster>::forecaster;
	using BaseOutForecast<BaseForecaster>::out_forecast;
	using BaseOutForecast<BaseForecaster>::lpl_record;
	using BaseOutForecast<BaseForecaster>::initialize;
	void initForecaster(LIST& fit_record) override {
		forecaster[0] = initialize_forecaster<BaseForecaster>(
			num_chains, lag, step, roll_y0[0], sparse, level,
			fit_record, seed_forecast, include_mean,
			stable_filter, nthreads, sv
		);
	}
	Eigen::MatrixXd buildDesign(int window) override {
		return build_x0(roll_mat[window], lag, include_mean);
	}
	void updateForecaster(RecordType& reg_record, int window, int chain) override {
		forecaster[window][chain] = std::make_unique<McmcVarForecaster<BaseForecaster>>(
			reg_record, step, roll_y0[window], lag, include_mean,
			stable_filter, static_cast<unsigned int>(seed_forecast[chain]), sv
		);
	}
};

template <template <typename> class BaseOutForecast = McmcRollforecastRun, typename BaseForecaster = RegForecaster>
class McmcVharforecastRun : public BaseOutForecast<BaseForecaster> {
public:
	McmcVharforecastRun(
		const Eigen::MatrixXd& y, int week, int month, int num_chains, int num_iter, int num_burn, int thin,
		bool sparse, double level, LIST& fit_record,
		LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
		const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		bool include_mean, bool stable, int step, const Eigen::MatrixXd& y_test, bool get_lpl,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, int nthreads, bool sv = true
	)
	: BaseOutForecast<BaseForecaster>(
			y, month, num_chains, num_iter, num_burn, thin, sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test, get_lpl,
			seed_chain, seed_forecast, nthreads, sv
		),
		har_trans(build_vhar(dim, week, month, include_mean)) {
		initialize(
			y, fit_record, param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, seed_chain
		);
	}
	virtual ~McmcVharforecastRun() = default;

protected:
	using typename BaseOutForecast<BaseForecaster>::BaseMcmc;
	using typename BaseOutForecast<BaseForecaster>::RecordType;
	using BaseOutForecast<BaseForecaster>::dim;
	using BaseOutForecast<BaseForecaster>::num_horizon;
	using BaseOutForecast<BaseForecaster>::step;
	using BaseOutForecast<BaseForecaster>::lag;
	using BaseOutForecast<BaseForecaster>::num_chains;
	using BaseOutForecast<BaseForecaster>::num_iter;
	using BaseOutForecast<BaseForecaster>::num_burn;
	using BaseOutForecast<BaseForecaster>::thin;
	using BaseOutForecast<BaseForecaster>::nthreads;
	using BaseOutForecast<BaseForecaster>::include_mean;
	using BaseOutForecast<BaseForecaster>::stable_filter;
	using BaseOutForecast<BaseForecaster>::sparse;
	using BaseOutForecast<BaseForecaster>::sv;
	using BaseOutForecast<BaseForecaster>::level;
	using BaseOutForecast<BaseForecaster>::seed_forecast;
	using BaseOutForecast<BaseForecaster>::roll_mat;
	using BaseOutForecast<BaseForecaster>::roll_y0;
	using BaseOutForecast<BaseForecaster>::model;
	using BaseOutForecast<BaseForecaster>::forecaster;
	using BaseOutForecast<BaseForecaster>::out_forecast;
	using BaseOutForecast<BaseForecaster>::lpl_record;
	using BaseOutForecast<BaseForecaster>::initialize;
	Eigen::MatrixXd har_trans;
	void initForecaster(LIST& fit_record) override {
		forecaster[0] = initialize_forecaster<BaseForecaster>(
			num_chains, lag, step, roll_y0[0], sparse, level,
			fit_record, seed_forecast, include_mean,
			stable_filter, nthreads, sv,
			har_trans
		);
	}
	Eigen::MatrixXd buildDesign(int window) override {
		return build_x0(roll_mat[window], lag, include_mean) * har_trans.transpose();
	}
	void updateForecaster(RecordType& reg_record, int window, int chain) override {
		forecaster[window][chain] = std::make_unique<McmcVharForecaster<BaseForecaster>>(
			reg_record, step, roll_y0[window], har_trans, lag, include_mean,
			stable_filter, static_cast<unsigned int>(seed_forecast[chain]), sv
		);
	}
};

} // namespace bvhar

#endif // BVHARFORECASTER_H