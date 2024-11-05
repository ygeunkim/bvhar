#ifndef SVFORECASTER_H
#define SVFORECASTER_H

#include "mcmcsv.h"

namespace bvhar {

class SvForecaster;
class SvVarForecaster;
class SvVharForecaster;
class SvVarSelectForecaster;
class SvVharSelectForecaster;

class SvForecaster {
public:
	SvForecaster(const SvRecords& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: sv_record(std::make_unique<SvRecords>(records)), rng(seed),
		response(response_mat), include_mean(include_mean), stable_filter(filter_stable), sv(sv),
		step(step), dim(response.cols()), var_lag(ord),
		dim_design(include_mean ? var_lag * dim + 1 : var_lag * dim),
		num_coef(sv_record->coef_record.cols()),
		num_alpha(include_mean ? num_coef - dim : num_coef), nrow_coef(num_alpha / dim),
		num_sim(sv_record->coef_record.rows()),
		last_pvec(Eigen::VectorXd::Zero(dim_design)),
		sv_update(Eigen::VectorXd::Zero(dim)),
		sv_sig(Eigen::VectorXd::Zero(dim)),
		post_mean(Eigen::VectorXd::Zero(dim)),
		predictive_distn(Eigen::MatrixXd::Zero(step, num_sim * dim)),
		coef_mat(Eigen::MatrixXd::Zero(num_coef / dim, dim)),
		contem_mat(Eigen::MatrixXd::Identity(dim, dim)),
		h_last_record(sv_record->lvol_record.rightCols(dim)),
		standard_normal(Eigen::VectorXd::Zero(dim)),
		tmp_vec(Eigen::VectorXd::Zero((var_lag - 1) * dim)),
		lpl(Eigen::VectorXd::Zero(step)) {
		last_pvec[dim_design - 1] = 1.0; // valid when include_mean = true
		last_pvec.head(var_lag * dim) = vectorize_eigen(response.colwise().reverse().topRows(var_lag).transpose().eval()); // [y_T^T, y_(T - 1)^T, ... y_(T - lag + 1)^T]
		// post_mean = last_pvec.head(dim); // y_T
		// tmp_vec = last_pvec.segment(dim, (var_lag - 1) * dim); // y_(T - 1), ... y_(T - lag + 1)
	}
	virtual ~SvForecaster() = default;
	Eigen::MatrixXd forecastDensity() {
		std::lock_guard<std::mutex> lock(mtx);
		// Eigen::VectorXd mean_draw = post_mean.replicate(num_sim, 1); // cbind(sims)
		// Eigen::MatrixXd predictive_distn(step, num_sim * dim); // rbind(step), cbind(sims)
		Eigen::VectorXd obs_vec = last_pvec; // y_T y_(T - 1), ... y_(T - lag + 1)
		for (int i = 0; i < num_sim; ++i) {
			last_pvec = obs_vec;
			post_mean = obs_vec.head(dim);
			tmp_vec = obs_vec.segment(dim, (var_lag - 1) * dim);
			updateParams(i);
			forecastOut(i);
		}
		return predictive_distn;
	}
	Eigen::MatrixXd forecastDensity(const Eigen::VectorXd& valid_vec) {
		std::lock_guard<std::mutex> lock(mtx);
		// Eigen::MatrixXd predictive_distn(step, num_sim * dim); // rbind(step), cbind(sims)
		Eigen::VectorXd obs_vec = last_pvec; // y_T y_(T - 1), ... y_(T - lag + 1)
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
	std::unique_ptr<SvRecords> sv_record;
	boost::random::mt19937 rng;
	std::mutex mtx;
	Eigen::MatrixXd response; // y0
	bool include_mean;
	bool stable_filter;
	bool sv;
	int step;
	int dim;
	int var_lag; // VAR order or month order of VHAR
	int dim_design;
	int num_coef;
	int num_alpha;
	int nrow_coef; // dim_design in VAR and dim_har in VHAR (without constant term)
	int num_sim;
	Eigen::VectorXd last_pvec; // [ y_(T + h - 1)^T, y_(T + h - 2)^T, ..., y_(T + h - p)^T, 1 ] (1 when constant term)
	Eigen::VectorXd sv_update; // h_(T + h)
	Eigen::VectorXd sv_sig; // sig_h
	Eigen::VectorXd post_mean; // posterior mean and y_(T + h - 1)
	Eigen::MatrixXd predictive_distn; // rbind(step), cbind(sims)
	Eigen::MatrixXd coef_mat; // include constant term when include_mean = true
	Eigen::MatrixXd contem_mat; // L
	Eigen::MatrixXd h_last_record; // h_T record
	Eigen::VectorXd standard_normal; // Z ~ N(0, I)
	Eigen::VectorXd tmp_vec; // y_(T + h - 2), ... y_(T + h - lag)
	Eigen::VectorXd lpl; // average log-predictive likelihood
	virtual void computeMean() = 0;
	void updateParams(int i) {
		coef_mat.topRows(nrow_coef) = unvectorize(sv_record->coef_record.row(i).head(num_alpha).transpose(), dim);
		if (include_mean) {
			coef_mat.bottomRows(1) = sv_record->coef_record.row(i).tail(dim);
		}
		// sv_update = h_last_record.row(i).transpose();
		// sv_sig = sv_record->lvol_sig_record.row(i).cwiseSqrt();
		sv_record->updateDiag(i, sv_update, sv_sig); // D^1/2
		contem_mat = build_inv_lower(dim, sv_record->contem_coef_record.row(i)); // L
	}
	void updateVariance() {
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
	void updateLpl(int h, const Eigen::VectorXd& valid_vec) {
		lpl[h] += sv_update.sum() / 2 - dim * log(2 * M_PI) / 2 - ((-sv_update / 2).array().exp() * (contem_mat * (post_mean - valid_vec)).array()).matrix().squaredNorm() / 2;
	}
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

class SvVarForecaster : public SvForecaster {
public:
	SvVarForecaster(const SvRecords& records, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: SvForecaster(records, step, response_mat, lag, include_mean, filter_stable, sv, seed) {
		if (stable_filter) {
			sv_record->subsetStable(num_alpha, 1.05);
			num_sim = sv_record->coef_record.rows();
			if (num_sim == 0) {
				STOP("No stable MCMC draws");
			}
		}
	}
	virtual ~SvVarForecaster() = default;

protected:
	void computeMean() override {
		post_mean = last_pvec.transpose() * coef_mat;
	}
};

class SvVharForecaster : public SvForecaster {
public:
	SvVharForecaster(const SvRecords& records, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: SvForecaster(records, step, response_mat, month, include_mean, filter_stable, sv, seed), har_trans(har_trans.sparseView()) {
		if (stable_filter) {
			sv_record->subsetStable(num_alpha, 1.05, har_trans.topLeftCorner(3 * dim, month * dim).sparseView());
			num_sim = sv_record->coef_record.rows();
			if (num_sim == 0) {
				STOP("No stable MCMC draws");
			}
		}
	}
	virtual ~SvVharForecaster() = default;

protected:
	// Eigen::MatrixXd har_trans;
	Eigen::SparseMatrix<double> har_trans;
	void computeMean() override {
		post_mean = last_pvec.transpose() * har_trans.transpose() * coef_mat;
	}
};

class SvVarSelectForecaster : public SvVarForecaster {
public:
	SvVarSelectForecaster(const SvRecords& records, double level, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: SvVarForecaster(records, step, response_mat, lag, include_mean, filter_stable, sv, seed),
		activity_graph(unvectorize(sv_record->computeActivity(level), dim)) {}
	SvVarSelectForecaster(const SvRecords& records, const Eigen::MatrixXd& selection, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: SvVarForecaster(records, step, response_mat, lag, include_mean, filter_stable, sv, seed), activity_graph(selection) {}
	virtual ~SvVarSelectForecaster() = default;

protected:
	void computeMean() override {
		post_mean = last_pvec.transpose() * (activity_graph.array() * coef_mat.array()).matrix();
	}

private:
	Eigen::MatrixXd activity_graph; // Activity graph computed after MCMC
};

class SvVharSelectForecaster : public SvVharForecaster {
public:
	SvVharSelectForecaster(const SvRecords& records, double level, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: SvVharForecaster(records, step, response_mat, har_trans, month, include_mean, filter_stable, sv, seed),
		activity_graph(unvectorize(sv_record->computeActivity(level), dim)) {}
	SvVharSelectForecaster(const SvRecords& records, const Eigen::MatrixXd& selection, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, bool filter_stable, bool sv, unsigned int seed)
	: SvVharForecaster(records, step, response_mat, har_trans, month, include_mean, filter_stable, sv, seed), activity_graph(selection) {}
	virtual ~SvVharSelectForecaster() = default;

protected:
	void computeMean() override {
		post_mean = last_pvec.transpose() * har_trans.transpose() * (activity_graph.array() * coef_mat.array()).matrix();
	}

private:
	Eigen::MatrixXd activity_graph; // Activity graph computed after MCMC
};

} // namespace bvhar

#endif // SVFORECASTER_H