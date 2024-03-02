#ifndef MINNFORECASTER_H
#define MINNFORECASTER_H

#include "minnesota.h"

namespace bvhar {

class MinnForecaster {
public:
	MinnForecaster(const MinnFit& fit, int step, const Eigen::MatrixXd& response_mat, int ord, bool include_mean)
	: response(response_mat),
		posterior_mean(fit._coef), posterior_prec(fit._prec),
		posterior_iw_scale(fit._iw_scale), posterior_iw_shape(fit._iw_shape),
		include_mean(include_mean), step(step), dim(posterior_mean.cols()), var_lag(ord),
		dim_design(include_mean ? var_lag * dim + 1 : var_lag * dim),
		pred_save(Eigen::MatrixXd::Zero(step, dim)),
		last_pvec(Eigen::VectorXd::Zero(dim_design)) {
		last_pvec[dim_design - 1] = 1.0; // valid when include_mean = true
		last_pvec.head(var_lag * dim) = response.colwise().reverse().topRows(var_lag).transpose().reshaped(); // [y_T^T, y_(T - 1)^T, ... y_(T - lag + 1)^T]
		point_forecast = last_pvec.head(dim); // y_T
		tmp_vec = last_pvec.segment(dim, (var_lag - 1) * dim); // y_(T - 1), ... y_(T - lag + 1)
	}
	virtual ~MinnForecaster() = default;
protected:
	Eigen::MatrixXd response;
	Eigen::MatrixXd posterior_mean;
	Eigen::MatrixXd posterior_prec;
	Eigen::MatrixXd posterior_iw_scale;
	double posterior_iw_shape;
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

} // namespace bvhar

#endif // MINNFORECASTER_H