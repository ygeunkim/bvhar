#ifndef SVFORECASTER_H
#define SVFORECASTER_H

#include "mcmcsv.h"

namespace bvhar {

class SvForecaster {
public:
	SvForecaster(const SvRecords& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean, unsigned int seed)
	: sv_record(records), rng(seed),
		response(response_mat), include_mean(include_mean),
		step(step), dim(response.cols()), var_lag(ord),
		dim_design(include_mean ? var_lag * dim + 1 : var_lag * dim),
		num_coef(sv_record.coef_record.cols()),
		num_alpha(include_mean ? num_coef - dim : num_coef), nrow_coef(num_alpha / dim),
		num_sim(sv_record.coef_record.rows()),
		last_pvec(Eigen::VectorXd::Zero(dim_design)),
		sv_update(Eigen::VectorXd::Zero(dim)),
		post_mean(Eigen::VectorXd::Zero(dim)),
		predictive_distn(Eigen::MatrixXd::Zero(step, num_sim * dim)),
		coef_mat(Eigen::MatrixXd::Zero(num_coef / dim, dim)),
		contem_mat(Eigen::MatrixXd::Identity(dim, dim)),
		h_last_record(sv_record.lvol_record.rightCols(dim)),
		standard_normal(Eigen::VectorXd::Zero(dim)) {
		last_pvec[dim_design - 1] = 1.0; // valid when include_mean = true
		last_pvec.head(var_lag * dim) = vectorize_eigen(response.colwise().reverse().topRows(var_lag).transpose().eval()); // [y_T^T, y_(T - 1)^T, ... y_(T - lag + 1)^T]
		point_forecast = last_pvec.head(dim); // y_T
		tmp_vec = last_pvec.segment(dim, (var_lag - 1) * dim); // y_(T - 1), ... y_(T - lag + 1)
	}
	virtual ~SvForecaster() = default;
	virtual void computeMean(int i) = 0;
	Eigen::MatrixXd forecastDensity() {
		std::lock_guard<std::mutex> lock(mtx);
		for (int h = 0; h < step; h++) {
			last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
			for (int i = 0; i < num_sim; i++) {
				coef_mat.topRows(nrow_coef) = unvectorize(sv_record.coef_record.row(i).head(num_alpha).transpose(), dim);
				if (include_mean) {
					coef_mat.bottomRows(1) = sv_record.coef_record.row(i).tail(dim);
				}
				last_pvec.head(dim) = point_forecast;
				computeMean(i); // can have overflow issue due to stability of each coef_record
				for (int j = 0; j < dim; j++) {
					standard_normal[j] = normal_rand(rng);
				}
				standard_normal.array() *= sv_record.lvol_sig_record.row(i).cwiseSqrt().array(); // sig_h Z ~ N(0, sig_h^2)
				sv_update = h_last_record.row(i).transpose() + standard_normal; // h_(t+1) = h_t + u_t
				h_last_record.row(i) = sv_update; // for next update
				contem_mat = build_inv_lower(dim, sv_record.contem_coef_record.row(i));
				for (int j = 0; j < dim; j++) {
					standard_normal[j] = normal_rand(rng);
				}
				standard_normal.array() *= (sv_update / 2).array().exp(); // D^(1/2) Z ~ N(0, D)
				point_forecast = post_mean + contem_mat.triangularView<Eigen::UnitLower>().solve(standard_normal); // N(post_mean, L^-1 D L)
				predictive_distn.block(h, i * dim, 1, dim) = point_forecast.transpose();
			}
			tmp_vec = last_pvec.head((var_lag - 1) * dim);
		}
		return predictive_distn;
	}
protected:
	SvRecords sv_record;
	boost::random::mt19937 rng;
	std::mutex mtx;
	Eigen::MatrixXd response; // y0
	bool include_mean;
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
	Eigen::VectorXd post_mean; // posterior mean
	Eigen::MatrixXd predictive_distn; // rbind(step), cbind(sims)
	Eigen::MatrixXd coef_mat; // include constant term when include_mean = true
	Eigen::MatrixXd contem_mat; // L
	Eigen::MatrixXd h_last_record; // h_T record
	Eigen::VectorXd standard_normal; // Z ~ N(0, I)
	Eigen::VectorXd point_forecast; // y_(T + h - 1)
	Eigen::VectorXd tmp_vec; // y_(T + h - 2), ... y_(T + h - lag)
};

class SvVarForecaster : public SvForecaster {
public:
	SvVarForecaster(const SvRecords& records, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean, unsigned int seed)
	: SvForecaster(records, step, response_mat, lag, include_mean, seed) {}
	virtual ~SvVarForecaster() = default;
	void computeMean(int i) override {
		post_mean = last_pvec.transpose() * coef_mat;
	}
};

class SvVharForecaster : public SvForecaster {
public:
	SvVharForecaster(const SvRecords& records, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean, unsigned int seed)
	: SvForecaster(records, step, response_mat, month, include_mean, seed), har_trans(har_trans) {}
	virtual ~SvVharForecaster() = default;
	void computeMean(int i) override {
		post_mean = last_pvec.transpose() * har_trans.transpose() * coef_mat;
	}
private:
	Eigen::MatrixXd har_trans;
};

} // namespace bvhar

#endif // SVFORECASTER_H