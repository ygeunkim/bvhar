#ifndef BVHARSIM_H
#define BVHARSIM_H

#include "bvharcommon.h"

Eigen::MatrixXd sim_mgaussian(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig);

Eigen::MatrixXd sim_mgaussian_chol(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig);

Eigen::MatrixXd sim_mgaussian_chol(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig, boost::random::mt19937& rng);

Eigen::MatrixXd sim_mstudent(int num_sim, double df, Eigen::VectorXd mu, Eigen::MatrixXd sig, int method);

Eigen::MatrixXd sim_matgaussian(Eigen::MatrixXd mat_mean, Eigen::MatrixXd mat_scale_u, Eigen::MatrixXd mat_scale_v);

Eigen::MatrixXd sim_iw(Eigen::MatrixXd mat_scale, double shape);

Rcpp::List sim_mniw(int num_sim, Eigen::MatrixXd mat_mean, Eigen::MatrixXd mat_scale_u, Eigen::MatrixXd mat_scale, double shape);

// Eigen::MatrixXd sim_wishart(Eigen::MatrixXd mat_scale, double shape);

namespace bvhar {

// Generate Lower Triangular Matrix of Wishart
// 
// This function generates \eqn{A = L (Q^{-1})^T}.
// 
// @param mat_scale Scale matrix of Wishart
// @param shape Shape of Wishart
// @details
// This function generates Wishart random matrix.
inline Eigen::MatrixXd sim_wishart(Eigen::MatrixXd mat_scale, double shape) {
  int dim = mat_scale.cols();
  if (shape <= dim - 1) {
    Rcpp::stop("Wrong 'shape'. shape > dim - 1 must be satisfied.");
  }
  if (mat_scale.rows() != mat_scale.cols()) {
    Rcpp::stop("Invalid 'mat_scale' dimension.");
  }
  if (dim != mat_scale.rows()) {
    Rcpp::stop("Invalid 'mat_scale' dimension.");
  }
  Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim);
  for (int i = 0; i < dim; i++) {
    mat_bartlett(i, i) = sqrt(bvhar::chisq_rand(shape - (double)i));
  }
  for (int i = 1; i < dim; i++) {
    for (int j = 0; j < i; j++) {
      mat_bartlett(i, j) = norm_rand();
    }
  }
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(mat_scale);
  Eigen::MatrixXd chol_scale = lltOfscale.matrixL();
  Eigen::MatrixXd chol_res = chol_scale * mat_bartlett;
  return chol_res * chol_res.transpose();
}

// Quasi-density of GIG
// 
// @param x postivie support
// @param lambda Index of modified Bessel function of third kind.
// @param beta Square of the multiplication of the other two parameters.
inline double dgig_quasi(double x, double lambda, double beta) {
	return pow(x, lambda - 1) * exp(-beta * (x + 1 / x) / 2);
}

// AR-Mehod for non-concave part
// 
// @param num_sim Number to generate process
// @param lambda Index of modified Bessel function of third kind.
// @param beta Square of the multiplication of the other two parameters.
inline Eigen::VectorXd rgig_nonconcave(int num_sim, double lambda, double beta) {
	Eigen::VectorXd res(num_sim);
	double mode = beta / (sqrt((1 - lambda) * (1 - lambda) + beta * beta) + 1 - lambda); // argmax of g(x)
	double x0, xstar, k1, k2, k3, A1, A2, A3;
	x0 = beta / (1 - beta); // subdomain (0, x0)
	xstar = std::max(x0, 2 / beta);
	k1 = dgig_quasi(mode, lambda, beta);
	A1 = k1 * x0;
	if (x0 < 2 / beta) { // subdomain (x0, 2 / beta)
		k2 = exp(-beta);
		if (lambda == 0) {
			A2 = k2 * log(2 / (beta * beta));
		} else {
			A2 = k2 * (pow(2 / beta, lambda) - pow(x0, lambda)) / lambda;
		}
	} else {
		k2 = 0;
		A2 = 0;
	}
	k3 = pow(xstar, lambda - 1);  // subdomain (xstar, inf)
	A3 = 2 * k3 * exp(-xstar * beta / 2) / beta;
	double A = A1 + A2 + A3;
	bool rejected;
	double draw_unif, draw_prop, cand, ar_const;
	// double draw_x, draw_y, cand; // bounded rectangle
	for (int i = 0; i < num_sim; i++) {
		rejected = true;
		while (rejected) {
			draw_unif = unif_rand(0, 1);
			draw_prop = unif_rand(0, A);
			if (draw_prop <= A1) {
				cand = x0 * draw_prop / A1;
				ar_const = k1;
			} else if (draw_prop <= A1 + A2) {
				draw_prop -= A1;
				cand = pow(pow(x0, lambda) + draw_prop * lambda / k2, 1 / lambda);
				ar_const = k2 * pow(cand, lambda - 1);
			} else {
				draw_prop -= A1 + A2;
				cand = -2 * log(exp(-xstar * beta / 2) - draw_prop * beta / (2 * k3));
				ar_const = k3 * exp(-cand * beta / 2);
			}
			rejected = draw_unif * ar_const > dgig_quasi(cand, lambda, beta);
		}
		res[i] = cand;
	}
	return res;
}

// Ratio-of-Uniforms without Mode Shift
// 
// @param num_sim Number to generate process
// @param lambda Index of modified Bessel function of third kind.
// @param beta Square of the multiplication of the other two parameters.
inline Eigen::VectorXd rgig_without_mode(int num_sim, double lambda, double beta) {
	Eigen::VectorXd res(num_sim);
	double arg_y = beta / (sqrt((1 - lambda) * (1 - lambda) + beta * beta) + 1 - lambda); // argmax of g(x)
	double arg_x = (1 + lambda + sqrt((1 + lambda) * (1 + lambda) + beta * beta)) / beta; // argmax of x g(x)
	double bound_y = sqrt(dgig_quasi(arg_y, lambda, beta)); // max sqrt(g(x))
	double bound_x = arg_x * sqrt(dgig_quasi(arg_x, lambda, beta)); // max x*sqrt(g(x))
	bool rejected;
	double draw_x, draw_y, cand; // bounded rectangle
	for (int i = 0; i < num_sim; i++) {
		rejected = true;
		while (rejected) {
			draw_x = unif_rand(0, bound_x);
			draw_y = unif_rand(0, bound_y);
			cand = draw_x / draw_y;
			rejected = draw_y * draw_y > dgig_quasi(cand, lambda, beta); // Check if U <= g(y) / unif(y)
		}
		res[i] = cand;
	}
	return res;
}

// Ratio-of-Uniforms with Mode Shift
// 
// @param num_sim Number to generate process
// @param lambda Index of modified Bessel function of third kind.
// @param beta Square of the multiplication of the other two parameters.
inline Eigen::VectorXd rgig_with_mode(int num_sim, double lambda, double beta) {
	Eigen::VectorXd res(num_sim);
	double arg_y = (sqrt((1 - lambda) * (1 - lambda) + beta * beta) - 1 + lambda) / beta; // argmax of g(x)
	double quad_coef = -2 * (lambda + 1) / beta - arg_y;
	double lin_coef = 2 * arg_y * (lambda - 1) / beta - 1;
	double p = lin_coef - quad_coef * quad_coef / 3;
	double q = 2 * quad_coef * quad_coef * quad_coef / 27 - quad_coef * lin_coef * arg_y / 3 + arg_y;
	double phi = acos(-q * sqrt(-27 / (p * p * p)) / 2);
	double arg_x_neg = sqrt(-p * 4 / 3) * cos(phi / 3 + M_PI * 4 / 3) - quad_coef / 3;
	double arg_x_pos = sqrt(-p * 4 / 3) * cos(phi / 3) - quad_coef / 3;
	double bound_y = sqrt(dgig_quasi(arg_y, lambda, beta));
	double bound_x_neg = (arg_x_neg - arg_y) * sqrt(dgig_quasi(arg_x_neg, lambda, beta));
	double bound_x_pos = (arg_x_pos - arg_y) * sqrt(dgig_quasi(arg_x_pos, lambda, beta));
	bool rejected;
	double draw_x, draw_y, cand; // bounded rectangle
	for (int i = 0; i < num_sim; i++) {
		rejected = true;
		while (rejected) {
			draw_x = unif_rand(bound_x_neg, bound_x_pos);
			draw_y = unif_rand(0, bound_y);
			cand = draw_x / draw_y + arg_y;
			rejected = draw_y * draw_y > dgig_quasi(cand, lambda, beta); // Check if U <= g(y) / unif(y)
		}
		res[i] = cand;
	}
	return res;
}

// Generate Generalized Inverse Gaussian Distribution
// 
// This function samples GIG(lambda, psi, chi) random variates.
// 
// @param num_sim Number to generate process
// @param lambda Index of modified Bessel function of third kind.
// @param psi Second parameter of GIG
// @param chi Third parameter of GIG
// 
// @references Hörmann, W., Leydold, J. Generating generalized inverse Gaussian random variates. Stat Comput 24, 547–557 (2014).
inline Eigen::VectorXd sim_gig(int num_sim, double lambda, double psi, double chi) {
	if (psi <= 0 || chi <= 0) {
		Rcpp::stop("Wrong 'psi' and 'chi' range.");
	}
	Eigen::VectorXd res(num_sim);
	double abs_lam = abs(lambda); // If lambda < 0, use 1 / X as the result
	double alpha = sqrt(psi / chi); // scaling parameter of quasi-density: scale the result
	double beta = sqrt(psi * chi); // second parameter of quasi-density
	double quasi_bound = sqrt(1 - lambda) * 2 / 3;
	if (abs_lam < 1 && beta <= quasi_bound) {
		res = rgig_nonconcave(num_sim, lambda, beta) * alpha; // non-T_(-1/2)-concave part
	} else if (abs_lam <= 1 && beta >= std::min(.5, quasi_bound) && beta <= 1) {
		res = rgig_without_mode(num_sim, lambda, beta) * alpha; // without mode shift
	} else if (abs_lam > 1 && beta > 1) {
		res = rgig_with_mode(num_sim, lambda, beta) * alpha; // with mode shift
	} else {
		Rcpp::stop("Wrong parameter ranges for quasi GIG density.");
	}
	if (lambda < 0) {
		res = res.cwiseInverse();
	}
	return res;
}

} //namespace bvhar

#endif // BVHARSIM_H
