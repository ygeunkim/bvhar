#ifndef BVHAR_BAYES_MNIW_TUNING_H
#define BVHAR_BAYES_MNIW_TUNING_H

#include "../../math/optim.h"
#include "../misc/minn_helper.h"
#include "../../math/design.h"

namespace baecon {
namespace bvhar {

class MinnesotaLogLik;
class MhMinnLogLik;

// param_vec: sigma(dim), lambda(1), daily(dim), weekly(dim), monthly(dim)
inline double minnesota_logml(const Eigen::VectorXd& param_vec,
															const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
															double eps, int lag, bool include_mean) {
	// int dim_design = x.cols();
	int num_design = y.rows();
	int dim = y.cols();
	Eigen::VectorXd sigma = param_vec.head(dim);
	double lam = param_vec[dim];
	Eigen::VectorXd daily = param_vec.segment(dim + 1, dim);
	Eigen::VectorXd weekly = Eigen::VectorXd::Zero(dim);
	Eigen::VectorXd monthly = Eigen::VectorXd::Zero(dim);
	if (param_vec.size() != 2 * dim + 1) {
		weekly = param_vec.segment(2 * dim + 1, dim);
		monthly = param_vec.segment(3 * dim + 1, dim);
	}
	Eigen::MatrixXd dummy_response = build_ydummy(lag, sigma, lam, daily, weekly, monthly, include_mean);
	Eigen::MatrixXd dummy_design = build_xdummy(
		Eigen::VectorXd::LinSpaced(lag, 1, lag),
		lam, sigma, eps, include_mean
	);
	// int num_dummy = dummy_design.rows();
	Eigen::MatrixXd prior_prec = dummy_design.transpose() * dummy_design;
	Eigen::MatrixXd prior_mean = prior_prec.selfadjointView<Eigen::Lower>().llt().solve(dummy_design.transpose() * dummy_response);
	Eigen::MatrixXd prior_scale = (dummy_response - dummy_design * prior_mean).transpose() * (dummy_response - dummy_design * prior_mean);
	// int prior_shape = num_dummy - dim_design + 2;
	// Eigen::MatrixXd ystar(num_design + num_dummy, dim);
	// Eigen::MatrixXd xstar(num_design + num_dummy, dim_design);
	// ystar << y,
	// 				 dummy_response;
	// xstar << x,
	// 				 dummy_design;
	Eigen::MatrixXd mn_prec = x.transpose() * x + prior_prec;
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec(mn_prec.selfadjointView<Eigen::Lower>());
	Eigen::MatrixXd coef = llt_of_prec.solve(x.transpose() * y);
	Eigen::MatrixXd x_prec = llt_of_prec.matrixU().solve<Eigen::OnTheRight>(x);
	Eigen::MatrixXd iw_scale = y.transpose() * (Eigen::MatrixXd::Identity(num_design, num_design) - x_prec * x_prec.transpose()) * y;
	int posterior_df = num_design - dim - 1;
	return compute_logml(dim, num_design, prior_prec, prior_scale, mn_prec, iw_scale, posterior_df);
}

// param_vec: psi(dim), lambda(1)
inline double minnesota_logml(const Eigen::VectorXd& param_vec, const Eigen::VectorXd& delta,
															const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
															double eps, int lag, bool include_mean) {
	int num_design = y.rows();
	int dim = y.cols();
	Eigen::VectorXd sigma = param_vec.head(dim);
	double lam = param_vec[dim];
	Eigen::VectorXd daily = delta.head(dim);
	Eigen::VectorXd weekly = Eigen::VectorXd::Zero(dim);
	Eigen::VectorXd monthly = Eigen::VectorXd::Zero(dim);
	if (delta.size() == 3 * dim) {
		weekly = delta.segment(dim, dim);
		monthly = delta.segment(2 * dim, dim);
	}
	Eigen::MatrixXd dummy_response = build_ydummy(lag, sigma, lam, daily, weekly, monthly, include_mean);
	Eigen::MatrixXd dummy_design = build_xdummy(
		Eigen::VectorXd::LinSpaced(lag, 1, lag),
		lam, sigma, eps, include_mean
	);
	// int num_dummy = dummy_design.rows();
	Eigen::MatrixXd prior_prec = dummy_design.transpose() * dummy_design;
	Eigen::MatrixXd prior_mean = prior_prec.selfadjointView<Eigen::Lower>().llt().solve(dummy_design.transpose() * dummy_response);
	Eigen::MatrixXd prior_scale = (dummy_response - dummy_design * prior_mean).transpose() * (dummy_response - dummy_design * prior_mean);
	Eigen::MatrixXd mn_prec = x.transpose() * x + prior_prec;
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec(mn_prec.selfadjointView<Eigen::Lower>());
	Eigen::MatrixXd coef = llt_of_prec.solve(x.transpose() * y);
	Eigen::MatrixXd x_prec = llt_of_prec.matrixU().solve<Eigen::OnTheRight>(x);
	Eigen::MatrixXd iw_scale = y.transpose() * (Eigen::MatrixXd::Identity(num_design, num_design) - x_prec * x_prec.transpose()) * y;
	int posterior_df = num_design - dim - 1;
	return compute_logml(dim, num_design, prior_prec, prior_scale, mn_prec, iw_scale, posterior_df);
}

class MinnesotaLogLik : public FuncMin {
public:
	MinnesotaLogLik(
		const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		int lag, bool include_mean,
		double eps = 1e-4
	)
	: design(x), response(y), lag(lag), eps(eps), include_mean(include_mean) {}
	virtual ~MinnesotaLogLik() = default;

	double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) override {
		double fx = minnesota_logml(x, design, response, eps, lag, include_mean);
		int param_size = x.size();
		double step_size = 1e-8;
		Eigen::VectorXd x_h(param_size);
		Eigen::VectorXd x_h_minus(param_size);
	#ifdef _OPENMP
		#pragma omp parallel for
	#endif
		for (int i = 0; i < param_size; ++i) {
			x_h = x;
			x_h_minus = x;
			x_h[i] += step_size;
			x_h_minus[i] -= step_size;
			double f_h = minnesota_logml(x_h, design, response, eps, lag, include_mean);
			double f_h_minus = minnesota_logml(x_h_minus, design, response, eps, lag, include_mean);
			grad[i] = (f_h - f_h_minus) / (2 * step_size);
		}
    return fx;
	}

private:
	Eigen::MatrixXd design, response;
	int lag;
	double eps;
	bool include_mean;
};

class MhMinnLogLik : public FuncMin {
public:
	MhMinnLogLik(
		const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		const Eigen::VectorXd& delta,
		int lag, bool include_mean,
		double eps = 1e-4
	)
	: design(x), response(y), delta(delta), lag(lag), eps(eps), include_mean(include_mean) {}
	virtual ~MhMinnLogLik() = default;

	double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) override {
		double fx = minnesota_logml(x, delta, design, response, eps, lag, include_mean);
		int param_size = x.size();
		double step_size = 1e-8;
		Eigen::VectorXd x_h(param_size);
		Eigen::VectorXd x_h_minus(param_size);
	#ifdef _OPENMP
		#pragma omp parallel for
	#endif
		for (int i = 0; i < param_size; ++i) {
			x_h = x;
			x_h_minus = x;
			x_h[i] += step_size;
			x_h_minus[i] -= step_size;
			double f_h = minnesota_logml(x_h, delta, design, response, eps, lag, include_mean);
			double f_h_minus = minnesota_logml(x_h_minus, delta, design, response, eps, lag, include_mean);
			grad[i] = (f_h - f_h_minus) / (2 * step_size);
		}
    return fx;
	}

private:
	Eigen::MatrixXd design, response;
	Eigen::VectorXd delta;
	int lag;
	double eps;
	bool include_mean;
};

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_MNIW_TUNING_H