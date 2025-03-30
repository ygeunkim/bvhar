#ifndef BVHAR_BAYES_MNIW_FORECASTER_H
#define BVHAR_BAYES_MNIW_FORECASTER_H

#include "./minnesota.h"
#include "../../math/random.h"

namespace bvhar {

class MinnForecaster;
class BvarForecaster;
class BvharForecaster;

class MinnForecaster {
public:
	MinnForecaster(const MinnFit& fit, int step, const Eigen::MatrixXd& response_mat, int ord, int num_sim, bool include_mean, unsigned int seed)
	: rng(seed), response(response_mat),
		posterior_mean(fit._coef), posterior_sig(fit._prec.inverse()),
		posterior_iw_scale(fit._iw_scale), posterior_iw_shape(fit._iw_shape),
	// MinnForecaster(const MinnRecords& records, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean)
	// : mn_record(records), response(response_mat),
		// posterior_mean(fit._coef), posterior_sig(fit._prec.inverse()),
		// posterior_iw_scale(fit._iw_scale), posterior_iw_shape(fit._iw_shape),
		include_mean(include_mean), step(step), dim(posterior_mean.cols()), var_lag(ord),
		// include_mean(include_mean), step(step), dim(response_mat.cols()), var_lag(ord),
		dim_design(include_mean ? var_lag * dim + 1 : var_lag * dim),
		num_sim(num_sim), coef_and_sig(num_sim, std::vector<Eigen::MatrixXd>(2)),
		// num_sim(mn_record.coef_record.rows()),
		// coef_mat(Eigen::MatrixXd::Zero(dim_design, dim)), cov_mat(Eigen::MatrixXd::Zero(dim, dim)),
		pred_save(Eigen::MatrixXd::Zero(step, dim)),
		mn_scl(Eigen::MatrixXd::Zero(1, 1)),
		density_forecast(step, num_sim * dim),
		predictive_distn(step, num_sim * dim),
		last_pvec(Eigen::VectorXd::Zero(dim_design)) {
		last_pvec[dim_design - 1] = 1.0; // valid when include_mean = true
		last_pvec.head(var_lag * dim) = response.colwise().reverse().topRows(var_lag).transpose().reshaped(); // [y_T^T, y_(T - 1)^T, ... y_(T - lag + 1)^T]
		point_forecast = last_pvec.head(dim); // y_T
		tmp_vec = last_pvec.segment(dim, (var_lag - 1) * dim); // y_(T - 1), ... y_(T - lag + 1)
	}
	virtual ~MinnForecaster() = default;
	virtual void updateVariance() = 0;
	virtual void computeMean() = 0;
	virtual void updateDensity(int h) = 0;
	// void updateDensity(int h) {
	// 	for (int i = 0; i < num_sim; ++i) {
	// 		coef_mat = unvectorize(mn_record.coef_record.row(i).transpose(), dim);
	// 		cov_mat = unvectorize(mn_record.sig_record.row(i).transpose(), dim);
	// 		last_pvec.head(dim) = point_forecast;
	// 		computeMean();
	// 		predictive_distn.block(h, i * dim, 1, dim) = point_forecast.transpose();
	// 	}
	// }
	void forecastPoint() {
		for (int h = 0; h < step; ++h) {
			last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
			last_pvec.head(dim) = point_forecast;
			computeMean();
			pred_save.row(h) = point_forecast;
			tmp_vec = last_pvec.head((var_lag - 1) * dim);
		}
	}
	void forecastDensity() {
	// Eigen::MatrixXd forecastDensity() {
		for (int i = 0; i < num_sim; ++i) {
			coef_and_sig[i] = sim_mn_iw(posterior_mean, posterior_sig, posterior_iw_scale, posterior_iw_shape, false, rng);
		}
		// std::lock_guard<std::mutex> lock(mtx);
		for (int h = 0; h < step; ++h) {
			last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
			// for (int i = 0; i < num_sim; ++i) {
			// 	coef_mat = unvectorize(mn_record.coef_record.row(i), dim);
			// 	cov_mat = unvectorize(mn_record.sig_record.row(i), dim);
			// 	last_pvec.head(dim) = point_forecast;
			// 	computeMean();
			// }
			last_pvec.head(dim) = point_forecast;
			updateVariance();
			computeMean();
			pred_save.row(h) = point_forecast;
			updateDensity(h);
			tmp_vec = last_pvec.head((var_lag - 1) * dim);
		}
		// return predictive_distn;
	}
	LIST returnForecast() const {
		return CREATE_LIST(
			NAMED("posterior_mean") = pred_save,
      NAMED("predictive") = predictive_distn
		);
	}
	Eigen::MatrixXd returnPoint() {
		forecastPoint();
		return pred_save;
	}
protected:
	// MinnRecords mn_record;
	// std::mutex mtx;
	BHRNG rng;
	Eigen::MatrixXd response;
	Eigen::MatrixXd posterior_mean;
	Eigen::MatrixXd posterior_sig;
	Eigen::MatrixXd posterior_iw_scale;
	double posterior_iw_shape;
	bool include_mean;
	int step;
	int dim;
	int var_lag; // VAR order or month order of VHAR
	int dim_design;
	int num_sim;
	std::vector<std::vector<Eigen::MatrixXd>> coef_and_sig; // (A, Sig) ~ MNIW
	// Eigen::MatrixXd coef_mat;
	// Eigen::MatrixXd cov_mat;
	Eigen::MatrixXd pred_save; // rbind(step)
	// Eigen::VectorXd sig_update; // se^2 for each forecast (except Sigma2 part, i.e. closed form)
	Eigen::MatrixXd mn_scl; // se^2 for each forecast (except Sigma2 part, i.e. closed form)
	Eigen::MatrixXd density_forecast;
	Eigen::MatrixXd predictive_distn;
	Eigen::VectorXd last_pvec; // [ y_(T + h - 1)^T, y_(T + h - 2)^T, ..., y_(T + h - p)^T, 1 ] (1 when constant term)
	Eigen::VectorXd point_forecast; // y_(T + h - 1)
	Eigen::VectorXd tmp_vec; // y_(T + h - 2), ... y_(T + h - lag)
};

class BvarForecaster : public MinnForecaster {
public:
	BvarForecaster(const MinnFit& fit, int step, const Eigen::MatrixXd& response_mat, int lag, int num_sim, bool include_mean, unsigned int seed)
	: MinnForecaster(fit, step, response_mat, lag, num_sim, include_mean, seed) {}
	// BvarForecaster(const MinnRecords& records, int step, const Eigen::MatrixXd& response_mat, int lag, bool include_mean)
	// : MinnForecaster(records, step, response_mat, lag, include_mean) {}
	virtual ~BvarForecaster() = default;
	void updateVariance() override {
		// sig_update[h] += last_pvec.transpose() * posterior_sig * last_pvec;
		mn_scl.array() = 1 + (last_pvec.transpose() * posterior_sig * last_pvec).array();
	}
	void computeMean() override {
		point_forecast = last_pvec.transpose() * posterior_mean;
		// point_forecast = last_pvec.transpose() * coef_mat;
	}
	void updateDensity(int h) override {
		for (int i = 0; i < num_sim; ++i) {
			density_forecast.block(h, i * dim, 1, dim) = last_pvec.transpose() * coef_and_sig[i][0];
      predictive_distn.block(h, i * dim, 1, dim) = sim_mn(
        density_forecast.block(h, i * dim, 1, dim),
        // Eigen::MatrixXd::Constant(1, 1, sig_update[h]),
				// Eigen::Map<Eigen::MatrixXd>(sig_update.block(h, 0, 1, 1).data(), 1, 1), // -> Matrix but too complex
				mn_scl,
				coef_and_sig[i][1],
				false,
				rng
      );
		}
	}
};

class BvharForecaster : public MinnForecaster {
public:
	BvharForecaster(const MinnFit& fit, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, int num_sim, bool include_mean, unsigned int seed)
	: MinnForecaster(fit, step, response_mat, month, num_sim, include_mean, seed), har_trans(har_trans),
		transformed_sig(har_trans.transpose() * posterior_sig * har_trans) {}
	// BvharForecaster(const MinnRecords& records, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, int month, bool include_mean)
	// : MinnForecaster(records, step, response_mat, month, include_mean), har_trans(har_trans) {}
		// transformed_sig(har_trans.transpose() * posterior_sig * har_trans) {}
	virtual ~BvharForecaster() = default;
	void updateVariance() override {
		// sig_update[h] += last_pvec.transpose() * transformed_sig * last_pvec;
		mn_scl.array() = 1 + (last_pvec.transpose() * transformed_sig * last_pvec).array();
	}
	void computeMean() override {
		point_forecast = last_pvec.transpose() * har_trans.transpose() * posterior_mean;
		// point_forecast = last_pvec.transpose() * har_trans.transpose() * coef_mat;
	}
	void updateDensity(int h) override {
		for (int i = 0; i < num_sim; ++i) {
			density_forecast.block(h, i * dim, 1, dim) = last_pvec.transpose() * har_trans.transpose() * coef_and_sig[i][0];
      predictive_distn.block(h, i * dim, 1, dim) = sim_mn(
        density_forecast.block(h, i * dim, 1, dim),
				// Eigen::Map<Eigen::MatrixXd>(sig_update.block(h, 0, 1, 1).data(), 1, 1), // -> Matrix but too complex
				// Eigen::MatrixXd::Constant(1, 1, sig_update[h]),
				mn_scl,
				coef_and_sig[i][1],
				false,
				rng
      );
		}
	}
private:
	Eigen::MatrixXd har_trans;
	Eigen::MatrixXd transformed_sig;
};

} // namespace bvhar

#endif // BVHAR_BAYES_MNIW_FORECASTER_H