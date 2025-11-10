#ifndef BVHAR_BAYES_OPTIM_H
#define BVHAR_BAYES_OPTIM_H

#include "../core/common.h"

namespace bvhar {

class FuncMin;
class OptimLbfgsb;

class FuncMin {
public:
	FuncMin() {}
	virtual ~FuncMin() = default;
	virtual double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) = 0;
};

class OptimLbfgsb {
public:
	OptimLbfgsb(
		std::unique_ptr<FuncMin>& log_lik,
		const Eigen::VectorXd& inits,
		double lower = .01, double upper = 10,
		const int max_iter = 300, const double& eps_f = 1e-6, const double& eps_g = 1e-5
	)
	: log_lik(std::move(log_lik)),
		param_size(inits.size()), param_vec(inits),
		lower_bound(Eigen::VectorXd::Constant(param_size, lower)),
		upper_bound(Eigen::VectorXd::Constant(param_size, upper)) {
		param.epsilon = eps_g;
		param.epsilon_rel = eps_g;
		param.past = 1;
		param.delta = eps_f;
		param.max_iterations = max_iter;
    param.max_linesearch = 100;
		// lbfgs_param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
		solver = std::make_unique<LBFGSpp::LBFGSBSolver<double>>(param);
	}

	OptimLbfgsb(
		std::unique_ptr<FuncMin>& log_lik,
		const Eigen::VectorXd& inits,
		const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
		const int max_iter = 300, const double& eps_f = 1e-6, const double& eps_g = 1e-5
	)
	: log_lik(std::move(log_lik)),
		param_size(inits.size()), param_vec(inits),
		lower_bound(lower), upper_bound(upper) {
		param.epsilon = eps_g;
		param.epsilon_rel = eps_g;
		param.past = 1;
		param.delta = eps_f;
		param.max_iterations = max_iter;
    param.max_linesearch = 100;
		// param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
		solver = std::make_unique<LBFGSpp::LBFGSBSolver<double>>(param);
	}

	virtual ~OptimLbfgsb() = default;
	
	void doOptim() {
		double fx;
		solver->minimize(*log_lik, param_vec, fx, lower_bound, upper_bound);
	}

	Eigen::VectorXd returnParams() {
		doOptim();
		return param_vec;
	}

private:
	std::unique_ptr<FuncMin> log_lik;
	int param_size;
	Eigen::VectorXd param_vec, lower_bound, upper_bound;
	LBFGSpp::LBFGSBParam<double> param;
	std::unique_ptr<LBFGSpp::LBFGSBSolver<double>> solver;
};

} // namespace bvhar

#endif // BVHAR_BAYES_OPTIM_H