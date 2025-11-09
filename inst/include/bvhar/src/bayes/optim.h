#ifndef BVHAR_BAYES_OPTIM_H
#define BVHAR_BAYES_OPTIM_H

#include "../core/common.h"

namespace bvhar {

class LogLik;
class OptimEmpBayes;

class LogLik {
public:
	LogLik() {}
	virtual ~LogLik() = default;
	virtual double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) = 0;
};

class OptimEmpBayes {
public:
	OptimEmpBayes(
		std::unique_ptr<LogLik>& log_lik,
		const Eigen::VectorXd& inits,
		double lower = .01, double upper = 10,
		const int max_iter = 300, const double& eps_f = 1e-6, const double& eps_g = 1e-5
	)
	: log_lik(std::move(log_lik)),
		param_size(inits.size()), param_vec(inits),
		lower_bound(Eigen::VectorXd::Constant(param_size, lower)),
		upper_bound(Eigen::VectorXd::Constant(param_size, upper)) {
		lbfgs_param.epsilon = eps_g;
		lbfgs_param.epsilon_rel = eps_g;
		lbfgs_param.past = 1;
		lbfgs_param.delta = eps_f;
		lbfgs_param.max_iterations = max_iter;
    lbfgs_param.max_linesearch = 100;
		// lbfgs_param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
		lbfgs_solver = std::make_unique<LBFGSpp::LBFGSBSolver<double>>(lbfgs_param);
	}

	OptimEmpBayes(
		std::unique_ptr<LogLik>& log_lik,
		const Eigen::VectorXd& inits,
		const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
		const int max_iter = 300, const double& eps_f = 1e-6, const double& eps_g = 1e-5
	)
	: log_lik(std::move(log_lik)),
		param_size(inits.size()), param_vec(inits),
		lower_bound(lower), upper_bound(upper) {
		lbfgs_param.epsilon = eps_g;
		lbfgs_param.epsilon_rel = eps_g;
		lbfgs_param.past = 1;
		lbfgs_param.delta = eps_f;
		lbfgs_param.max_iterations = max_iter;
    lbfgs_param.max_linesearch = 100;
		lbfgs_param.max_linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
		lbfgs_solver = std::make_unique<LBFGSpp::LBFGSBSolver<double>>(lbfgs_param);
	}

	virtual ~OptimEmpBayes() = default;
	
	void doOptim() {
		double fx;
		int niter = lbfgs_solver->minimize(*log_lik, param_vec, fx, lower_bound, upper_bound);
	}

	Eigen::VectorXd returnParams() {
		doOptim();
		return param_vec;
	}

private:
	std::unique_ptr<LogLik> log_lik;
	int param_size;
	Eigen::VectorXd param_vec, lower_bound, upper_bound;
	LBFGSpp::LBFGSBParam<double> lbfgs_param;
	std::unique_ptr<LBFGSpp::LBFGSBSolver<double>> lbfgs_solver;
};

} // namespace bvhar

#endif // BVHAR_BAYES_OPTIM_H