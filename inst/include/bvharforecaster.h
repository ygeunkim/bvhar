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
	LIST coef_list = fit_record[coef_name];
	LIST a_list = fit_record[a_name];
	LIST d_list = fit_record["d_record"];
	if (include_mean) {
		LIST c_list = fit_record[c_name];
		record = std::make_unique<LdltRecords>(
			CAST<Eigen::MatrixXd>(coef_list[CAST_INT(chain_id)]),
			CAST<Eigen::MatrixXd>(c_list[CAST_INT(chain_id)]),
			CAST<Eigen::MatrixXd>(a_list[CAST_INT(chain_id)]),
			CAST<Eigen::MatrixXd>(d_list[CAST_INT(chain_id)])
		);
	} else {
		record = std::make_unique<LdltRecords>(
			CAST<Eigen::MatrixXd>(coef_list[CAST_INT(chain_id)]),
			CAST<Eigen::MatrixXd>(a_list[CAST_INT(chain_id)]),
			CAST<Eigen::MatrixXd>(d_list[CAST_INT(chain_id)])
		);
	}
}

inline void initialize_record(std::unique_ptr<SvRecords>& record, int chain_id, LIST& fit_record, bool include_mean, STRING& coef_name, STRING& a_name, STRING& c_name) {
	LIST coef_list = fit_record[coef_name];
	LIST a_list = fit_record[a_name];
	LIST h_list = fit_record["h_record"];
	LIST sigh_list = fit_record["sigh_record"];
	if (include_mean) {
		LIST c_list = fit_record[c_name];
		record = std::make_unique<SvRecords>(
			CAST<Eigen::MatrixXd>(coef_list[CAST_INT(chain_id)]),
			CAST<Eigen::MatrixXd>(c_list[CAST_INT(chain_id)]),
			CAST<Eigen::MatrixXd>(h_list[CAST_INT(chain_id)]),
			CAST<Eigen::MatrixXd>(a_list[CAST_INT(chain_id)]),
			CAST<Eigen::MatrixXd>(sigh_list[CAST_INT(chain_id)])
		);
	} else {
		record = std::make_unique<SvRecords>(
			CAST<Eigen::MatrixXd>(coef_list[CAST_INT(chain_id)]),
			CAST<Eigen::MatrixXd>(h_list[CAST_INT(chain_id)]),
			CAST<Eigen::MatrixXd>(a_list[CAST_INT(chain_id)]),
			CAST<Eigen::MatrixXd>(sigh_list[CAST_INT(chain_id)])
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
	bool activity = (level != 0); // Optional<double> level = NULLOPT
	if (!sparse && activity) {
		STOP("If 'level != 0', 'spare' should be true.");
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

template <typename BaseForecaster = RegForecaster>
class McmcForecastRun {
public:
	McmcForecastRun(
		int num_chains, int lag, int step, const Eigen::MatrixXd& response_mat,
		bool sparse, double level, LIST& fit_record,
		const Eigen::VectorXi& seed_chain, bool include_mean, bool stable, int nthreads,
		bool sv = true
	)
	: num_chains(num_chains), nthreads(nthreads), density_forecast(num_chains) {
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

} // namespace bvhar

#endif // BVHARFORECASTER_H