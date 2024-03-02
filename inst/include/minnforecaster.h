#ifndef MINNFORECASTER_H
#define MINNFORECASTER_H

#include "minnesota.h"
#include "bvharsim.h"

namespace bvhar {

class MinnForecaster {
public:
	MinnForecaster(const MinnFit& fit, int step, const Eigen::MatrixXd& response_mat, int ord, int num_sim, bool include_mean)
	: response(response_mat),
		posterior_mean(fit._coef), posterior_sig(fit._prec.inverse()),
		posterior_iw_scale(fit._iw_scale), posterior_iw_shape(fit._iw_shape),
		include_mean(include_mean), step(step), dim(posterior_mean.cols()), var_lag(ord),
		dim_design(include_mean ? var_lag * dim + 1 : var_lag * dim),
		num_sim(num_sim), coef_and_sig(num_sim, std::vector<Eigen::MatrixXd>(2)),
		pred_save(Eigen::MatrixXd::Zero(step, dim)),
		sig_update(Eigen::VectorXd::Ones(step)),
		density_forecast(step, num_sim * dim), predictive_distn(step, num_sim * dim),
		last_pvec(Eigen::VectorXd::Zero(dim_design)) {
		last_pvec[dim_design - 1] = 1.0; // valid when include_mean = true
		last_pvec.head(var_lag * dim) = response.colwise().reverse().topRows(var_lag).transpose().reshaped(); // [y_T^T, y_(T - 1)^T, ... y_(T - lag + 1)^T]
		point_forecast = last_pvec.head(dim); // y_T
		tmp_vec = last_pvec.segment(dim, (var_lag - 1) * dim); // y_(T - 1), ... y_(T - lag + 1)
	}
	virtual ~MinnForecaster() = default;
	virtual void updateVariance(int i) = 0;
	virtual void computeMean(int i) = 0;
	virtual void updateDensity(int i) = 0;
	void forecastDensity() {
		for (int i = 0; i < num_sim; ++i) {
			coef_and_sig[i] = sim_mn_iw(posterior_mean, posterior_sig, posterior_iw_scale, posterior_iw_shape);
		}
		for (int i = 0; i < step; ++i) {
			last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
			last_pvec.head(dim) = point_forecast;
			updateVariance(i);
			computeMean(i);
			updateDensity(i);
			tmp_vec = last_pvec.head((var_lag - 1) * dim);
		}
	}
protected:
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
	Eigen::MatrixXd pred_save; // rbind(step)
	Eigen::VectorXd sig_update; // se^2 for each forecast (except Sigma2 part, i.e. closed form)
	Eigen::MatrixXd density_forecast;
	Eigen::MatrixXd predictive_distn;
	Eigen::VectorXd last_pvec; // [ y_(T + h - 1)^T, y_(T + h - 2)^T, ..., y_(T + h - p)^T, 1 ] (1 when constant term)
	Eigen::VectorXd point_forecast; // y_(T + h - 1)
	Eigen::VectorXd tmp_vec; // y_(T + h - 2), ... y_(T + h - lag)
};

class BvarForecaster : public MinnForecaster {
public:
	BvarForecaster(const MinnFit& fit, int step, const Eigen::MatrixXd& response_mat, int lag, int num_sim, bool include_mean)
	: MinnForecaster(fit, step, response_mat, lag, num_sim, include_mean) {}
	virtual ~BvarForecaster() = default;
	void updateVariance(int i) override {
		sig_update[i] += last_pvec.transpose() * posterior_sig * last_pvec;
	}
	void computeMean(int i) override {
		point_forecast = last_pvec.transpose() * posterior_mean;
	}
	void updateDensity(int i) override {
		for (int b = 0; b < num_sim; ++b) {
			density_forecast.block(i, b * dim, 1, dim) = last_pvec * coef_and_sig[b][0];
      predictive_distn.block(i, b * dim, 1, dim) = sim_matgaussian(
        density_forecast.block(i, b * dim, 1, dim),
        // sig_update[i].reshaped(1, 1),
				Eigen::Map<Eigen::MatrixXd>(sig_update.block(i, 0, 1, 1).data(), 1, 1), // -> Matrix but too complex
				coef_and_sig[b][1]
      );
		}
	}
};

} // namespace bvhar

#endif // MINNFORECASTER_H