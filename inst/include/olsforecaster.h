#ifndef OLSFORECASTER_H
#define OLSFORECASTER_H

#include "bvharcommon.h"
#include "ols.h"

namespace bvhar {

class OlsForecaster {
public:
	OlsForecaster(const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, bool include_mean)
	: response(response_mat), coef_mat(fit._coef),
		include_mean(include_mean), step(step), dim(coef_mat.cols()), var_lag(fit._ord),
		// dim_design(coef_mat.rows()),
		dim_design(include_mean ? var_lag * dim + 1 : var_lag * dim),
		pred_save(Eigen::MatrixXd::Zero(step, dim)),
		last_pvec(Eigen::VectorXd::Zero(dim_design)) {
		last_pvec[dim_design - 1] = 1.0; // valid when include_mean = true
		last_pvec.head(var_lag * dim) = vectorize_eigen(response.colwise().reverse().topRows(var_lag).transpose().eval()); // [y_T^T, y_(T - 1)^T, ... y_(T - lag + 1)^T]
		point_forecast = last_pvec.head(dim); // y_T
		tmp_vec = last_pvec.segment(dim, (var_lag - 1) * dim); // y_(T - 1), ... y_(T - lag + 1)
	}
	virtual ~OlsForecaster() = default;
	virtual void updatePred() = 0;
	Eigen::MatrixXd forecastPoint() {
		for (int h = 0; h < step; ++h) {
			last_pvec.segment(dim, (var_lag - 1) * dim) = tmp_vec;
			last_pvec.head(dim) = point_forecast;
			updatePred();
			pred_save.row(h) = point_forecast.transpose();
			tmp_vec = last_pvec.head((var_lag - 1) * dim);
		}
		return pred_save;
	}
protected:
	Eigen::MatrixXd response;
	Eigen::MatrixXd coef_mat;
	bool include_mean;
	int step;
	int dim;
	int var_lag; // VAR order or month order of VHAR
	int dim_design;
	Eigen::MatrixXd pred_save; // rbind(step)
	Eigen::VectorXd last_pvec; // [ y_(T + h - 1)^T, y_(T + h - 2)^T, ..., y_(T + h - p)^T, 1 ] (1 when constant term)
	Eigen::VectorXd point_forecast; // y_(T + h - 1)
	Eigen::VectorXd tmp_vec; // y_(T + h - 2), ... y_(T + h - lag)
};

class VarForecaster : public OlsForecaster {
public:
	VarForecaster(const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, bool include_mean)
	: OlsForecaster(fit, step, response_mat, include_mean) {}
	virtual ~VarForecaster() = default;
	void updatePred() override {
		point_forecast = last_pvec.transpose() * coef_mat; // y(T + h)^T = [yhat(T + h - 1)^T, ..., yhat(T + 1)^T, y(T)^T, ..., y(T + h - lag)^T] * Ahat
	}
};

class VharForecaster : public OlsForecaster {
public:
	VharForecaster(const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, bool include_mean)
	: OlsForecaster(fit, step, response_mat, include_mean), har_trans(har_trans) {}
	virtual ~VharForecaster() = default;
	void updatePred() override {
		point_forecast = last_pvec.transpose() * har_trans.transpose() * coef_mat; // y(T + h)^T = [yhat(T + h - 1)^T, ..., yhat(T + 1)^T, y(T)^T, ..., y(T + h - lag)^T] * C(HAR) * Ahat
	}
private:
	Eigen::MatrixXd har_trans;
};

} // namespace bvhar

#endif // OLSFORECASTER_H